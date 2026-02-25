# POME — Pipeline for Observation in Mid-infrared with ERIS

POME is a lightweight, script-based reduction pipeline for ERIS mid-IR imaging observations, supporting:
- Vortex coronagraphic observations (`vortex`)
- vAPP coronagraphic observations (`coro_app`)
- Imaging in pupil-stabilized mode (`img_pupil`)

The main entry point is `pome.py`, which orchestrates the execution of the required calibration and reduction scripts.

---

## Requirements

Tested as a collection of standalone Python scripts.

Suggested environment:
- Python 3.9+ (recommended)
- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `opencv-python`
- `tqdm`
- `imutils`
- `scikit-image` (used by some utilities)
- `vip_hci` (only if you use `save_cube.py`)

Install (example with pip):
```bash
pip install numpy scipy astropy matplotlib opencv-python tqdm imutils scikit-image vip_hci
```

## Repository Layout

The pipeline assumes the following directory structure:

<repo_root>/  
│  
├── pome.py  
├── eris_calib_dark.py  
├── eris_calib_dark_flat.py  
├── eris_calib_dark_psf.py  
├── eris_calib_flat.py  
├── eris_elab_psf.py  
├── eris_elab_sky.py  
├── eris_elab_obj.py  
├── eris_elab_img_pupil.py  
├── eris_background_img_pupil.py  
├── eris_align.py  
├── eris_check_frames.py  
├── save_cube.py  
├── vapp_eris_background.py  
├── vapp_eris_elab.py  
├── vapp_eris_dh_estraction.py  
│  
├── raw/  
│ └── <TARGETNAME>/  
│ ├── calib/  
│ │ ├── dark_obj/  
│ │ ├── dark_flat/  
│ │ ├── dark_psf/  
│ │ └── flat/  
│ │  
│ ├── object/  
│ ├── psf/  
│ ├── sky/  
│ │  
│ ├── listA.txt  
│ ├── listB.txt  
│ ├── obj_list.txt  
│ └── sky_list.txt  
│  
└── reduced/  
└── <TARGETNAME>/  
├── calib/  
├── sky/  
├── object/  
├── psf/  
├── clean/  
└── image/  


### Notes

- `<TARGETNAME>` must match the name passed to `pome.py`.
- The `raw/` directory contains the original FITS files and frame lists.
- The `reduced/` directory is automatically populated by the pipeline.
- Some scripts expect auxiliary files (e.g. `master_bpm_lamp.fits`) to be available at the repository root or in a consistent path.
- The pipeline relies on hard-coded relative paths. Moving scripts into subdirectories may require modifying `pome.py`.
