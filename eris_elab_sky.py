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
import glob
import cv2
import os
from tqdm import tqdm
from astropy.visualization import ZScaleInterval
import sys

if len(sys.argv) != 3:
    print("Usage: python eris_calib_dark_flat_w4.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

##Definizione delle costanti
dim=256

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

def lista_file_fits(directory):
    # Ottieni la lista dei file nella directory
    lista_file = os.listdir(directory)

    # Filtra i file con estensione '.fits'
    file_fits = [file for file in lista_file if file.endswith('.fits')]

    return file_fits
    
def read_acq1_keywords(fits_file):
    """
    Reads the HIERARCH ESO DET ACQ1 WIN NY and WIN STRY keywords from a FITS header.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.

    Returns
    -------
    tuple
        A tuple (nx, stry) where:
        - nx is the value of 'HIERARCH ESO DET ACQ1 WIN NY'
        - stry is the value of 'HIERARCH ESO DET ACQ1 WIN STRY'
    """
    with fits.open(fits_file) as hdul:
        header = hdul[0].header
        nx = header.get('HIERARCH ESO DET ACQ1 WIN NY')
        stry = header.get('HIERARCH ESO DET ACQ1 WIN STRY')
    return nx, stry
    
##Apro Sky
File_sky = np.loadtxt(PATH+'raw/'+received_variable+'/'+'sky_list.txt', dtype=str, unpack=True)
nx, stry=read_acq1_keywords(PATH+'raw/'+received_variable+'/sky/'+File_sky[0]+'.fits')

##Apro flat    
flat = ApriFits(PATH+'reduced/'+received_variable+'/calib/flat/master_flat/master_flat.fits')
Flat=np.array(flat)
Flat=Flat[stry:stry+nx,0:2048]

##Apro Science dark
dark_sc = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_obj/master_dark/master_dark.fits')
Master_Dark_Science=np.array(dark_sc)

##Apro Flat dark
dark_fl = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_flat/master_dark_flat/master_dark.fits')
Master_Dark_Flat=np.array(dark_fl)
Master_Dark_Flat=Master_Dark_Flat[stry:stry+nx,0:2048]

##Apro Maschera
mask = ApriFits(PATH+'reduced/'+received_variable+'/calib/bad_pixel_mask.fits')
mask_inv=np.array(mask)

##Apro Maschera ESO 
mask_ESO = ApriFits(PATH+'master_bpm_lamp.fits')
mask_ESO[mask_ESO>0] = 10000
mask_ESO[mask_ESO==0] = 1
mask_ESO[mask_ESO==10000] = 0
mask_ESO=np.array(mask_ESO)
mask_ESO=mask_ESO[stry:stry+nx,0:2048]

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
media=np.nanmedian(Flat_corrected) ##FAI CON LA MEDIANA E MEDIA
Flat_norm=Flat_corrected/media


for File in tqdm(File_sky, desc='Elaborazione'):
    ##Science
    Science = ApriFits(PATH+'raw/'+received_variable+'/sky/'+File+'.fits')
    ScienceA = []
    
    
    for i in range(0,len(Science)):
        Light=Science[i]
        Light=Light*mask_inv
        Light[Light == 0] = 'nan'
        Light=Light-Dark_Science
        Light=Light/Flat_norm
        mask = np.isnan(Light)
        Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
    
        ScienceA.append(Light)
        
    ##Salvo i fits
    ScienceA=np.array(ScienceA)
    ScienceA=np.median(ScienceA, axis=0)
    folder_name = "media"
    os.makedirs(PATH+'reduced/'+received_variable+'/sky/'+folder_name, exist_ok=True)
    print(f"Directory '{folder_name}' created (or already exists).")
    fits.writeto(PATH+'reduced/'+received_variable+'/sky/media/'+File+'_sky.fits', ScienceA)


##Plot
fig2 = plt.figure()
ax1 = plt.gca()
z  = ZScaleInterval()
z1, z2 = z.get_limits(ScienceA)
plt.imshow(ScienceA, cmap=cm.hot, vmin = z1*2, vmax = z2*2)
ax1.invert_yaxis()
ax1.set_title('Sky frame')
ax1.set_xlabel('Pixel',fontsize=15)
ax1.set_ylabel('Pixel',fontsize=15)
plt.grid(None)
plt.colorbar(label='$counts$ $[a.u.]$')

plt.savefig(PATH+"reduced/"+received_variable+"/image/sky.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

