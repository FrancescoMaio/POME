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

##Apro flat    
flat = ApriFits(PATH+'reduced/'+received_variable+'/calib/flat/master_flat/master_flat.fits')
Flat=np.array(flat)

##Apro Science dark
dark_sc = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_obj/master_dark/master_dark.fits')
Master_Dark_Science=np.array(dark_sc)

##Apro Flat dark
dark_fl = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_flat/master_dark_flat/master_dark.fits')
Master_Dark_Flat=np.array(dark_fl)

##Apro Maschera
mask = ApriFits(PATH+'reduced/'+received_variable+'/calib/bad_pixel_mask.fits')
mask_inv=np.array(mask)

##Apro Maschera ESO 
mask_ESO = ApriFits(PATH+'master_bpm_lamp.fits')
mask_ESO[mask_ESO>0] = 10000
mask_ESO[mask_ESO==0] = 1
mask_ESO[mask_ESO==10000] = 0
mask_ESO=np.array(mask_ESO)
mask_ESO=mask_ESO[0:2048-2,0:2048]


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

File_A, File_B, xa,ya,xb,yb = np.loadtxt('list.txt', dtype=str, unpack=True)

folder_name = "sky"
os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")

for i in tqdm(range(0,len(File_A)), desc='Background generation'):

    all_coorB=(xb[i],yb[i])
    all_coorA=(xa[i],ya[i])

    ##Science
    Science1 = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+FileB[i]+'Master_flatB.fits')
    
    ##Science1
    Science = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+FileA[i]+'Master_flatA.fits')
    
    Master_flat_B = []
    media_B=[]
    std1_B=[]
    std2_B=[]
    for i in range(0,len(Science)):
        print(i)
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
    
    fits.writeto(PATH+'reduced/'+received_variable+'/sky/'+File1+'Master_flatA.fits', Master_flat_B)
    fits.writeto(PATH+'reduced/'+received_variable+'/sky/'+File2+'Master_flatB.fits', Master_flat_A)
    np.savetxt(PATH+'reduced/'+received_variable+'/sky/'+File1+'info_flatA.txt', np.transpose([np.linspace(0,40, 40),media_B,std1_B,std2_B]))
    np.savetxt(PATH+'reduced/'+received_variable+'/sky/'+File2+'info_flatB.txt', np.transpose([np.linspace(0,40, 40),media_A,std1_A,std2_A]))
