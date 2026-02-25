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
import imutils
import glob
import cv2
import os
import sys
from tqdm import tqdm
from astropy.visualization import ZScaleInterval

if len(sys.argv) != 3:
    print("Usage: python vapp_eris_background.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

##Definizione delle costanti
dim=350

def ApriFits(percorso):
    image_file = get_pkg_data_filename(percorso)
    fits.info(image_file)
    immagine = fits.getdata(image_file, ext=0)
    return immagine

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

##Apro science
File_A, xa,ya = np.loadtxt(PATH+'raw/'+received_variable+'/listA.txt', dtype=str, unpack=True)
File_B, xb,yb = np.loadtxt(PATH+'raw/'+received_variable+'/listB.txt', dtype=str, unpack=True)
nx, stry=read_acq1_keywords(PATH+'raw/'+received_variable+'/object/'+File_A[0]+'.fits')

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


folder_name = "sky"
os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")

for j in tqdm(range(0,len(File_A)), desc='Background generation'):

    cent_coorB=(int(yb[j]),int(xb[j]))
    cent_coorA=(int(ya[j]),int(xa[j]))
    ##Science
    Science1 = ApriFits(PATH+'raw/'+received_variable+'/object/'+File_B[j]+'.fits')
    
    ##Science1
    Science = ApriFits(PATH+'raw/'+received_variable+'/object/'+File_A[j]+'.fits')
    
    Master_flat_B = []
    media_B=[]
    std1_B=[]
    std2_B=[]
    for i in range(0,len(Science)):
        Flat_s=Science[i]
        Flat_s=Flat_s*mask_inv
        Flat_s[Flat_s == 0] = 'nan'
        Flat_s=Flat_s-Dark_Science
        Flat_s=Flat_s/Flat_norm
        mask = np.isnan(Flat_s)
        Flat_s[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Flat_s[~mask])
    
        Flat_s=Flat_s[cent_coorB[0]-dim:cent_coorB[0]+dim,cent_coorB[1]-dim:cent_coorB[1]+dim]
        Master_flat_B.append(Flat_s)
        med=np.nanmean(Flat_s[524:662,496:659])
        std1=np.nanstd(Flat_s[524:662,496:659])
        std2=np.nanstd(Flat_s[113:162,389:440])
        media_B.append(med)
        std1_B.append(std1)
        std2_B.append(std2)
    
    Master_flat_A = []
    media_A=[]
    std1_A=[]
    std2_A=[]
    
    for i in range(0,len(Science)):
        Flat_s=Science1[i]
        Flat_s=Flat_s*mask_inv
        Flat_s[Flat_s == 0] = 'nan'
        Flat_s=Flat_s-Dark_Science
        Flat_s=Flat_s/Flat_norm
        mask = np.isnan(Flat_s)
        Flat_s[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Flat_s[~mask])
    
        Flat_s=Flat_s[cent_coorA[0]-dim:cent_coorA[0]+dim,cent_coorA[1]-dim:cent_coorA[1]+dim]
        Master_flat_A.append(Flat_s)
        med=np.nanmean(Flat_s[524:662,496:659])
        std1=np.nanstd(Flat_s[524:662,496:659])
        std2=np.nanstd(Flat_s[113:162,389:440])
        media_A.append(med)
        std1_A.append(std1)
        std2_A.append(std2)
    
    Master_flat_B=np.array(Master_flat_B)
    Master_flat_B=np.mean(Master_flat_B, axis=0)
    media_B=np.array(media_B)
    std1_B=np.array(std1_B)
    std2_B=np.array(std2_B)
    
    Master_flat_A=np.array(Master_flat_A)
    Master_flat_A=np.mean(Master_flat_A, axis=0)
    media_A=np.array(media_A)
    std1_A=np.array(std1_A)
    std2_A=np.array(std2_A)

    fits.writeto(PATH+'reduced/'+received_variable+'/sky/'+File_A[j]+'Master_flatA.fits', Master_flat_B)
    fits.writeto(PATH+'reduced/'+received_variable+'/sky/'+File_B[j]+'Master_flatB.fits', Master_flat_A)
    np.savetxt(PATH+'reduced/'+received_variable+'/sky/'+File_A[j]+'info_flatA.txt', np.transpose([np.linspace(0,len(Science), len(Science)),media_B,std1_B,std2_B]))
    np.savetxt(PATH+'reduced/'+received_variable+'/sky/'+File_B[j]+'info_flatB.txt', np.transpose([np.linspace(0,len(Science), len(Science)),media_A,std1_A,std2_A]))
