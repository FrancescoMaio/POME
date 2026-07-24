#!/usr/bin/env python3
"""Riduzione batch delle PSF ERIS.

Il file PSF_LIST_PATH deve contenere, per ogni riga:

    object_fits   sky_fits   x_approssimativa   y_approssimativa

Le righe vuote e quelle che iniziano con '#' vengono ignorate.
I nomi dei file FITS possono essere forniti con o senza estensione .fits.

Esecuzione:

    python reduce_psf.py /data/ERIS nome_dataset

Il percorso della lista PSF viene impostato direttamente nella costante
PSF_LIST_PATH.
"""

import argparse
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from scipy.optimize import OptimizeWarning, curve_fit


# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

# Modificare questo percorso con il percorso reale del file della lista PSF.

SPOT_GUESSES = (
    (1130, 50),
    (670, 161),
    (1235, 230),
)

SPOT_SEARCH_BOX = 50
SPOT_FIT_HALF_SIZE = 30
LARGE_CUTOUT_HALF_SIZE = 200
FINAL_CUTOUT_HALF_SIZE = 80
PSF_MASK_RADIUS = 150


logging.basicConfig(
    filename="fit_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pulisce una serie di PSF object usando gli sky associati."
    )

    parser.add_argument(
        "path",
        help="Radice dei dati, ad esempio /data/ERIS/",
    )

    parser.add_argument(
        "dataset",
        help="Nome della cartella/dataset osservativo",
    )

    parser.add_argument(
        "--no-blob",
        action="store_true",
        help=(
            "Allinea lo sky all'object tramite gli spot prima della sottrazione. "
            "Senza questa opzione viene mantenuto il comportamento blob=True "
            "del codice originale."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sovrascrive i FITS di output già esistenti.",
    )

    return parser.parse_args()

def ensure_fits_suffix(name):
    """Aggiunge l'estensione .fits se non è già presente."""
    return name if name.lower().endswith(".fits") else f"{name}.fits"


def read_input_table(filename):
    """Legge la lista object, sky e coordinate approssimative della PSF."""
    entries = []

    with open(filename, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            columns = stripped.split()

            if len(columns) != 4:
                raise ValueError(
                    f"Riga {line_number}: attese 4 colonne, "
                    f"trovate {len(columns)}: {stripped}"
                )

            object_name, sky_name, x_text, y_text = columns

            try:
                x_psf = float(x_text)
                y_psf = float(y_text)
            except ValueError as exc:
                raise ValueError(
                    f"Riga {line_number}: coordinate non valide: "
                    f"{x_text}, {y_text}"
                ) from exc

            entries.append(
                (
                    ensure_fits_suffix(object_name),
                    ensure_fits_suffix(sky_name),
                    x_psf,
                    y_psf,
                )
            )

    if not entries:
        raise ValueError("Il file di input non contiene righe valide.")

    return entries


def read_fits(path):
    """Legge il primary HDU di un file FITS come array float64."""
    return np.asarray(fits.getdata(path, ext=0), dtype=np.float64)


def read_acq1_keywords(fits_file):
    """Legge le dimensioni della finestra ACQ1 dall'header FITS."""
    with fits.open(fits_file) as hdul:
        header = hdul[0].header

        ny = header.get("HIERARCH ESO DET ACQ1 WIN NY")
        stry = header.get("HIERARCH ESO DET ACQ1 WIN STRY")

    if ny is None or stry is None:
        raise KeyError(
            f"Keyword ACQ1 WIN NY/STRY mancanti nell'header di {fits_file}"
        )

    return int(ny), int(stry)


def two_d_gaussian(
    xy,
    amplitude,
    xo,
    yo,
    sigma_x,
    sigma_y,
    theta,
    offset,
):
    """Modello gaussiano bidimensionale ellittico e ruotato."""
    x, y = xy

    a = (
        np.cos(theta) ** 2 / (2 * sigma_x**2)
        + np.sin(theta) ** 2 / (2 * sigma_y**2)
    )

    b = (
        -np.sin(2 * theta) / (4 * sigma_x**2)
        + np.sin(2 * theta) / (4 * sigma_y**2)
    )

    c = (
        np.sin(theta) ** 2 / (2 * sigma_x**2)
        + np.cos(theta) ** 2 / (2 * sigma_y**2)
    )

    model = offset + amplitude * np.exp(
        -(
            a * (x - xo) ** 2
            + 2 * b * (x - xo) * (y - yo)
            + c * (y - yo) ** 2
        )
    )

    return model.ravel()


def interpolate_nans(array):
    """Sostituisce NaN e valori infiniti tramite interpolazione lineare."""
    result = np.asarray(array, dtype=np.float64).copy()
    invalid = ~np.isfinite(result)

    if not invalid.any():
        return result

    if invalid.all():
        raise ValueError("L'immagine contiene solo valori non finiti.")

    flat = result.ravel()
    bad = ~np.isfinite(flat)
    good_indices = np.flatnonzero(~bad)

    flat[bad] = np.interp(
        np.flatnonzero(bad),
        good_indices,
        flat[~bad],
    )

    return flat.reshape(result.shape)


def translate_image(image, dx, dy):
    """Trasla un'immagine di dx e dy pixel."""
    height, width = image.shape

    matrix = np.float32(
        [
            [1, 0, dx],
            [0, 1, dy],
        ]
    )

    return cv2.warpAffine(
        image.astype(np.float64),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def find_max_position_in_subframe(frame, coor, box_size=50):
    """Trova il massimo locale attorno a una posizione approssimativa."""
    x = int(round(coor[0]))
    y = int(round(coor[1]))

    half = box_size // 2

    y0 = max(0, y - half)
    y1 = min(frame.shape[0], y + half + 1)

    x0 = max(0, x - half)
    x1 = min(frame.shape[1], x + half + 1)

    subframe = frame[y0:y1, x0:x1]

    if subframe.size == 0 or not np.isfinite(subframe).any():
        raise ValueError(
            f"Nessun pixel valido vicino a ({x}, {y})."
        )

    iy, ix = np.unravel_index(
        np.nanargmax(subframe),
        subframe.shape,
    )

    return x0 + ix, y0 + iy


def extract_cutout(image, center, half_size):
    """Estrae un cutout quadrato attorno a center."""
    x, y = center

    x = int(round(x))
    y = int(round(y))

    x0 = x - half_size
    x1 = x + half_size

    y0 = y - half_size
    y1 = y + half_size

    if (
        x0 < 0
        or y0 < 0
        or x1 > image.shape[1]
        or y1 > image.shape[0]
    ):
        raise ValueError(
            f"Cutout {2 * half_size}x{2 * half_size} fuori dal frame "
            f"attorno a ({x}, {y})."
        )

    return image[y0:y1, x0:x1].copy()


def fit_centroid(cutout, label, initial_amplitude=None):
    """Determina il centro tramite fit gaussiano bidimensionale."""
    data = interpolate_nans(cutout)

    height, width = data.shape
    yy, xx = np.mgrid[:height, :width]

    background = float(np.nanmedian(data))
    peak = float(np.nanmax(data))

    if initial_amplitude is None:
        amplitude = max(peak - background, 1.0)
    else:
        amplitude = initial_amplitude

    p0 = (
        amplitude,
        width / 2,
        height / 2,
        5.0,
        5.0,
        0.0,
        background,
    )

    bounds = (
        (
            0.0,
            0.0,
            0.0,
            0.3,
            0.3,
            -np.pi,
            -np.inf,
        ),
        (
            np.inf,
            width - 1.0,
            height - 1.0,
            width,
            height,
            np.pi,
            np.inf,
        ),
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)

            popt, _ = curve_fit(
                two_d_gaussian,
                np.vstack(
                    (
                        xx.ravel(),
                        yy.ravel(),
                    )
                ),
                data.ravel(),
                p0=p0,
                bounds=bounds,
                maxfev=20000,
            )

        dx = popt[1] - width / 2
        dy = popt[2] - height / 2

        return dx, dy

    except (OptimizeWarning, RuntimeError, ValueError) as exc:
        logging.error(
            "Fit non convergente per %s: %s",
            label,
            exc,
        )

        return np.nan, np.nan


def calibrate_cube(cube, mask, dark_science, flat_norm):
    """Applica maschera, dark e flat e media il cubo lungo l'asse temporale."""
    calibrated = cube * mask

    calibrated[calibrated == 0] = np.nan
    calibrated = (calibrated - dark_science) / flat_norm
    calibrated = interpolate_nans(calibrated)

    return np.nanmean(calibrated, axis=0)


def measure_spot_offsets(image, guesses, prefix):
    """Misura gli offset sub-pixel degli spot."""
    offsets = []

    for index, guess in enumerate(guesses, start=1):
        position = find_max_position_in_subframe(
            image,
            guess,
            SPOT_SEARCH_BOX,
        )

        cutout = extract_cutout(
            image,
            position,
            SPOT_FIT_HALF_SIZE,
        )

        dx, dy = fit_centroid(
            cutout,
            f"{prefix} spot {index}",
        )

        offsets.append((dx, dy))

    return np.asarray(offsets, dtype=float)


def mask_out_psf(image, center, radius=PSF_MASK_RADIUS):
    """Maschera con NaN la regione centrale contenente la PSF."""
    yy, xx = np.ogrid[
        : image.shape[0],
        : image.shape[1],
    ]

    cx, cy = center

    outside = (
        (xx - cx) ** 2 + (yy - cy) ** 2
    ) > radius**2

    result = image.copy()
    result[~outside] = np.nan

    return result


def clean_stripes(psf):
    """Rimuove pattern additivi per riga e per colonna."""
    center = (
        (psf.shape[1] - 1) / 2,
        (psf.shape[0] - 1) / 2,
    )

    background = mask_out_psf(
        psf,
        center,
    )

    row_medians = np.nanmedian(
        background,
        axis=1,
    )

    corrected = psf - row_medians[:, np.newaxis]

    background = mask_out_psf(
        corrected,
        center,
    )

    column_medians = np.nanmedian(
        background,
        axis=0,
    )

    return corrected - column_medians[np.newaxis, :]


def load_calibrations(root, dataset, reference_object):
    """Carica e prepara flat, dark e bad-pixel mask."""
    raw_psf_dir = root / "raw" / dataset / "psf"
    reduced_dir = root / "reduced" / dataset

    reference_path = raw_psf_dir / reference_object

    ny, stry = read_acq1_keywords(reference_path)

    flat = read_fits(
        reduced_dir
        / "calib"
        / "flat"
        / "master_flat"
        / "master_flat.fits"
    )
    flat = flat[stry : stry + ny, :2048]

    dark_science = read_fits(
        reduced_dir
        / "calib"
        / "dark_psf"
        / "master_dark"
        / "master_dark.fits"
    )

    dark_flat = read_fits(
        reduced_dir
        / "calib"
        / "dark_flat"
        / "master_dark_flat"
        / "master_dark.fits"
    )
    dark_flat = dark_flat[stry : stry + ny, :2048]

    bad_pixel_mask = read_fits(
        reduced_dir
        / "calib"
        / "bad_pixel_mask.fits"
    )

    eso_mask = read_fits(
        root / "master_bpm_lamp.fits"
    )

    eso_mask = (eso_mask == 0).astype(np.float64)
    eso_mask = eso_mask[stry : stry + ny, :2048]

    combined_mask = bad_pixel_mask * eso_mask

    masked_flat = flat * combined_mask
    masked_dark_flat = dark_flat * combined_mask
    masked_dark_science = dark_science * combined_mask

    masked_flat[masked_flat == 0] = np.nan
    masked_dark_flat[masked_dark_flat == 0] = np.nan
    masked_dark_science[masked_dark_science == 0] = np.nan

    flat_corrected = masked_flat - masked_dark_flat
    normalization = np.nanmedian(flat_corrected)

    if not np.isfinite(normalization) or normalization == 0:
        raise ValueError(
            "Normalizzazione del flat non valida."
        )

    flat_norm = flat_corrected / normalization

    return (
        raw_psf_dir,
        reduced_dir,
        combined_mask,
        masked_dark_science,
        flat_norm,
    )


def process_pair(
    object_path,
    sky_path,
    approximate_position,
    mask,
    dark_science,
    flat_norm,
    align_sky,
):
    """Calibra una coppia object/sky e produce la PSF finale."""
    sky_cube = read_fits(sky_path)
    object_cube = read_fits(object_path)

    sky = calibrate_cube(
        sky_cube,
        mask,
        dark_science,
        flat_norm,
    )

    obj = calibrate_cube(
        object_cube,
        mask,
        dark_science,
        flat_norm,
    )

    if align_sky:
        sky_offsets = measure_spot_offsets(
            sky,
            SPOT_GUESSES,
            sky_path.name,
        )

        obj_offsets = measure_spot_offsets(
            obj,
            SPOT_GUESSES,
            object_path.name,
        )

        dx = (
            np.nanmedian(obj_offsets[:, 0])
            - np.nanmedian(sky_offsets[:, 0])
        )

        dy = (
            np.nanmedian(obj_offsets[:, 1])
            - np.nanmedian(sky_offsets[:, 1])
        )

        if not np.isfinite(dx) or not np.isfinite(dy):
            raise RuntimeError(
                "Impossibile determinare l'offset object-sky dagli spot."
            )

        sky = translate_image(
            sky,
            dx,
            dy,
        )

    psf = obj - sky

    coarse_center = find_max_position_in_subframe(
        psf,
        approximate_position,
        box_size=50,
    )

    psf = extract_cutout(
        psf,
        coarse_center,
        LARGE_CUTOUT_HALF_SIZE,
    )

    psf = clean_stripes(psf)

    local_center = (
        LARGE_CUTOUT_HALF_SIZE,
        LARGE_CUTOUT_HALF_SIZE,
    )

    central_cutout = extract_cutout(
        psf,
        local_center,
        SPOT_FIT_HALF_SIZE,
    )

    dx, dy = fit_centroid(
        central_cutout,
        f"PSF centrale {object_path.name}",
    )

    if not np.isfinite(dx) or not np.isfinite(dy):
        raise RuntimeError(
            "Fit finale della PSF non convergente."
        )

    psf = translate_image(
        psf,
        -dx,
        -dy,
    )

    return extract_cutout(
        psf,
        local_center,
        FINAL_CUTOUT_HALF_SIZE,
    )


def output_name_for(object_name):
    """Mantiene il nome originale del file object."""
    return Path(object_name).name


def main():
    args = parse_args()
    
    PSF_LIST_PATH = Path(args.path+'raw/'+args.dataset+'/psf/psf_list.txt')

    root = Path(args.path).expanduser().resolve()
    psf_list_path = PSF_LIST_PATH.expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(
            f"La radice dei dati non esiste: {root}"
        )

    if not psf_list_path.is_file():
        raise FileNotFoundError(
            f"File della lista PSF non trovato: {psf_list_path}"
        )

    entries = read_input_table(psf_list_path)

    (
        raw_psf_dir,
        reduced_dir,
        mask,
        dark_science,
        flat_norm,
    ) = load_calibrations(
        root,
        args.dataset,
        entries[0][0],
    )

    output_dir = reduced_dir / "psf"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []

    for index, (
        object_name,
        sky_name,
        x_psf,
        y_psf,
    ) in enumerate(entries, start=1):

        object_path = raw_psf_dir / object_name
        sky_path = raw_psf_dir / sky_name
        output_path = output_dir / output_name_for(object_name)

        print(
            f"[{index}/{len(entries)}] "
            f"{object_name} - sky: {sky_name}"
        )

        try:
            if not object_path.is_file():
                raise FileNotFoundError(
                    f"Object non trovato: {object_path}"
                )

            if not sky_path.is_file():
                raise FileNotFoundError(
                    f"Sky non trovato: {sky_path}"
                )

            if output_path.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Il file di output esiste già: {output_path}. "
                    "Usare --overwrite per sovrascriverlo."
                )

            cleaned_psf = process_pair(
                object_path=object_path,
                sky_path=sky_path,
                approximate_position=(x_psf, y_psf),
                mask=mask,
                dark_science=dark_science,
                flat_norm=flat_norm,
                align_sky=args.no_blob,
            )

            fits.writeto(
                output_path,
                cleaned_psf.astype(np.float32),
                overwrite=args.overwrite,
            )

            print(f"    salvata: {output_path}")

        except Exception as exc:
            logging.exception(
                "Errore durante la riduzione di %s",
                object_name,
            )

            failures.append(
                (
                    object_name,
                    str(exc),
                )
            )

            print(f"    ERRORE: {exc}")

    completed = len(entries) - len(failures)

    print(
        f"\nCompletate: {completed}/{len(entries)}"
    )

    if failures:
        print("File non processati:")

        for name, message in failures:
            print(f"  - {name}: {message}")

        print(
            "\nI dettagli completi degli errori sono disponibili "
            "nel file fit_errors.log."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()