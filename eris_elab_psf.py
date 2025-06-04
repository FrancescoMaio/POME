from matplotlib import pyplot as plt, cm
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from matplotlib import colors
from scipy.stats import norm
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm
import scipy.optimize as opt
from matplotlib.patches import Arc
import numpy as np
#import imutils
import glob
import cv2
import os
import sys
from astropy.visualization import ZScaleInterval
import logging

# Set up logging
log_file = "fit_errors.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s - %(message)s")

if len(sys.argv) != 3:
    print("Usage: python eris_elab_psf.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

##Definizione delle costanti
dim=350
center = (200, 463)
spot1=(1150,50)
spot2=(690,161)
spot3=(1255,230)

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def ApriFits(percorso):
    image_file = get_pkg_data_filename(percorso)
    fits.info(image_file)
    immagine = fits.getdata(image_file, ext=0)
    return immagine

def translate_image(image, dx, dy):
    # Ottenere le dimensioni dell'immagine
    height, width = image.shape[:2]

    # Definire la matrice di trasformazione per la traslazione
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    # Applicare la trasformazione all'immagine utilizzando la funzione warpAffine
    translated_image = cv2.warpAffine(image, M, (width, height))

    return translated_image
    
def mask_psf(image, shape, coor):
    mask_1 = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask_1, coor, 150, 255, thickness=-1)
    
    mask_inv = mask_1
    mask_inv[mask_inv==0] = 1
    mask_inv[mask_inv==255] = 0

    masked_image=image*mask_inv
    masked_image[masked_image == 0] = 'nan'
    return masked_image, mask_1

def find_max_position_in_subframe(frame, coor=(1184,414), box_size=50):
    """
    Find the position of the maximum value within a 50x50 subframe centered at the specified coordinates.
    
    Parameters:
        frame (2D np.ndarray): The image/frame in which to search.
        coor (tuple): Center coordinates (y, x).
        box_size (int): Size of the square subframe (default is 50).
    
    Returns:
        max_pos (tuple): Global coordinates (y, x) of the maximum value in the original frame.
    """
    x, y = coor
    half_size = box_size // 2
    
    # Define the bounds of the subframe, ensuring they don't go outside the image
    y_start = max(0, y - half_size)
    y_end = min(frame.shape[0], y + half_size)
    x_start = max(0, x - half_size)
    x_end = min(frame.shape[1], x + half_size)
    
    # Extract the subframe
    subframe = frame[y_start:y_end, x_start:x_end]
    # Find the index of the maximum value within the subframe
    max_idx = np.unravel_index(np.argmax(subframe), subframe.shape)
    
    # Convert the subframe index to the global frame coordinates
    max_pos = (x_start + max_idx[1],y_start + max_idx[0])
    
    return max_pos

##Apro flat    
flat = ApriFits(PATH+'reduced/'+received_variable+'/calib/flat/master_flat/master_flat.fits')
Flat=np.array(flat)
Flat=Flat[1154:1154+512,0:2048]

##Apro Science dark
dark_sc = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_psf/master_dark/master_dark.fits')
Master_Dark_Science=np.array(dark_sc) #[404:404+512,0:2048]

##Apro Flat dark
dark_fl = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_flat/master_dark_flat/master_dark.fits')
Master_Dark_Flat=np.array(dark_fl)
Master_Dark_Flat=Master_Dark_Flat[1154:1154+512,0:2048]

##Apro Maschera
mask = ApriFits(PATH+'reduced/'+received_variable+'/calib/bad_pixel_mask.fits')
mask_inv=np.array(mask)

##Apro Maschera ESO 
mask_ESO = ApriFits(PATH+'master_bpm_lamp.fits')
mask_ESO[mask_ESO>0] = 10000
mask_ESO[mask_ESO==0] = 1
mask_ESO[mask_ESO==10000] = 0
mask_ESO=np.array(mask_ESO)
mask_ESO=mask_ESO[1154:1154+512,0:2048]

mask_inv=mask_inv*mask_ESO

##Applico la maschera
Dark_Flat=Master_Dark_Flat*mask_inv
Dark_Flat[Dark_Flat == 0] = 'nan'

Dark_Science=Master_Dark_Science*mask_inv
Dark_Science[Dark_Science == 0] = 'nan'

Flat=Flat*mask_inv
Flat[Flat == 0] = 'nan'

##Correggo il flat
Flat_corrected=Flat-Dark_Flat
media=np.nanmedian(Flat_corrected)
Flat_norm=Flat_corrected/media

##PSF_Sky
PSF_Sky = ApriFits(PATH+'raw/'+received_variable+'/psf/psf_sky.fits')
Light=PSF_Sky
Light=Light*mask_inv
Light[Light == 0] = 'nan'
Light=Light-Dark_Science
Light=Light/Flat_norm
mask = np.isnan(Light)
Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
PSF_Sky=Light
spot1=find_max_position_in_subframe(PSF_Sky, spot1, 50)
spot2=find_max_position_in_subframe(PSF_Sky, spot2, 50)
spot3=find_max_position_in_subframe(PSF_Sky, spot3, 50)
spot1S=PSF_Sky[spot1[1]-30:spot1[1]+30,spot1[0]-30:spot1[0]+30]
spot2S=PSF_Sky[spot2[1]-30:spot2[1]+30,spot2[0]-30:spot2[0]+30]
spot3S=PSF_Sky[spot3[1]-30:spot3[1]+30,spot3[0]-30:spot3[0]+30]

##PSF
PSF = ApriFits(PATH+'raw/'+received_variable+'/psf/psf_obj.fits')
Light=PSF
Light=Light*mask_inv
Light[Light == 0] = 'nan'
Light=Light-Dark_Science
Light=Light/Flat_norm
mask = np.isnan(Light)
Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
PSF=Light

spot1L=PSF[spot1[1]-30:spot1[1]+30,spot1[0]-30:spot1[0]+30]
spot2L=PSF[spot2[1]-30:spot2[1]+30,spot2[0]-30:spot2[0]+30]
spot3L=PSF[spot3[1]-30:spot3[1]+30,spot3[0]-30:spot3[0]+30]

x = np.linspace(0, 59, 60)
y = np.linspace(0, 59, 60)
X,Y = np.meshgrid(x,y)
xdata = np.vstack((X.ravel(),Y.ravel()))
mask = np.isnan(spot1L)
spot1L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot1L[~mask])

immagine_flat = spot1L.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 1L")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan) 
dx1L=popt[1]-30
dy1L=popt[2]-30
mask = np.isnan(spot1S)
spot1S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot1S[~mask])
immagine_flat = spot1S.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 1S")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan) 
dx1S=popt[1]-30
dy1S=popt[2]-30

immagine_flat = spot2L.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 2L")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan)
dx1L=popt[1]-30
dx2L=popt[1]-30
dy2L=popt[2]-30
mask = np.isnan(spot2S)
spot2S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot2S[~mask])
immagine_flat = spot2S.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 2S")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan)
dx2S=popt[1]-30
dy2S=popt[2]-30

mask = np.isnan(spot3L)
spot3L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot3L[~mask])
immagine_flat = spot3L.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 3L")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan)
dx3L=popt[1]-30
dy3L=popt[2]-30
mask = np.isnan(spot3S)
spot3S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot3S[~mask])
immagine_flat = spot3S.flatten()
try:
    popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
except (opt.OptimizeWarning, RuntimeError) as e:
    error_message = f"Fit did not converge: {str(e)}"
    logging.error(error_message+" Spot 3S")

    # Assign NaN values on failure
    popt = np.full(7, np.nan)  
    pcov = np.full((7, 7), np.nan)
dx3S=popt[1]-30
dy3S=popt[2]-30

medianxS=np.median([dx1S,dx2S,dx3S])
medianyS=np.median([dy1S,dy2S,dy3S])
medianxL=np.median([dx1L,dx2L,dx3L])
medianyL=np.median([dy1L,dy2L,dy3L])

offx=medianxL-medianxS
offy=medianyL-medianyS

PSF_Sky1 = translate_image(PSF_Sky.astype(np.float64), offx, offy)
PSF=PSF-PSF_Sky1
print(PSF)

coor=find_max_position_in_subframe(PSF)
##correggo bande
PSF=PSF[coor[1]-250:coor[1]+250,coor[0]-250:coor[0]+250]

img_array, test=mask_psf(PSF, PSF.shape, coor)
row_means = np.nanmedian(img_array, axis=1)
PSF= PSF-row_means[:, np.newaxis]
img_array1, test=mask_psf(PSF, PSF.shape, coor)
column_medians = np.nanmedian(img_array1, axis=0)
PSF= PSF-column_medians


spot1L=PSF[250-30:250+30,250-30:250+30]

x = np.linspace(0, 59, 60)
y = np.linspace(0, 59, 60)
X,Y = np.meshgrid(x,y)
xdata = np.vstack((X.ravel(),Y.ravel()))
mask = np.isnan(spot1L)
spot1L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot1L[~mask])
immagine_flat = spot1L.flatten()
popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(10000,30,30,5,5,0,1.2e4))
dx=popt[1]-30
dy=popt[2]-30

PSF = translate_image(PSF.astype(np.float64), -dx, -dy)
PSF=PSF[250-80:250+80,250-80:250+80]

os.makedirs(PATH+'reduced/'+received_variable+'/psf', exist_ok=True)
print(f"Directory 'PSF' created (or already exists).")
fits.writeto(PATH+'reduced/'+received_variable+'/psf/psf.fits', PSF)


