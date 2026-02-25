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
from scipy.ndimage import maximum_filter
from matplotlib.patches import Arc
import numpy as np
from skimage.feature import peak_local_max
import imutils
import glob
import cv2
from astropy.visualization import ZScaleInterval
import os
import sys
from tqdm import tqdm
import logging

# Set up logging
log_file = "fit_errors.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s - %(message)s")

if len(sys.argv) != 3:
    print("Usage: python eris_elab_img_pupil.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

##Definizione delle costanti
dim=230

def mask_psf(image, shape, coor):
    mask_1 = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask_1, coor, 60, 255, thickness=-1)
    mask_inv = mask_1
    mask_inv[mask_inv==0] = 1
    mask_inv[mask_inv==255] = 0

    masked_image=image*mask_inv
    masked_image[masked_image == 0] = 'nan'
    return masked_image, mask_1

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

# Check if arrays have the same length
if len(File_A) != len(File_B):
    error_message = f"Error: Mismatch in number of files. File_A={len(File_A)}, File_B={len(File_B)}"
    print(error_message)
    logging.error(error_message)
    sys.exit(1)  # Exit the program with error code 1

folder_name = "clean"
os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")

for j in tqdm(range(0,len(File_A)), desc='Background remuval'):

    all_coorB=(int(xb[j]),int(yb[j]))
    all_coorA=(int(xa[j]),int(ya[j]))
    ##Master_flat_A
    Master_flat_B = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+File_A[j]+'_Master_flatB.fits')
    
    ##Master_flat_B
    Master_flat_A = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+File_B[j]+'_Master_flatA.fits')
    
    ##Science
    Science = ApriFits(PATH+'raw/'+received_variable+'/object/'+File_B[j]+'.fits')
    
    Flat_s=Science[1]
    Flat_s=Flat_s*mask_inv
    Flat_s[Flat_s == 0] = 'nan'
    Flat_s=Flat_s-Dark_Science
    
    lim_inf=270
    lim_sup=1910
    
    ##Science1
    Science1 = ApriFits(PATH+'raw/'+received_variable+'/object/'+File_A[j]+'.fits')
    ScienceA = []
    dxA = []
    dyA = []
    sigmaA = []
    ScienceB = []
    dxB = []
    dyB = []
    sigmaB = []

    no_backgroundA=[]
    no_backgroundB=[]
    
    for i in range(0,len(Science1)):
        Light=Science[i]
        Light=Light*mask_inv
        Light[Light == 0] = 'nan'
        Light=Light-Dark_Science
        Light=Light/Flat_norm
        mask = np.isnan(Light)
        Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
        B=(Light[all_coorB[0]-dim:all_coorB[0]+dim,all_coorB[1]-dim:all_coorB[1]+dim]-Master_flat_A)
        img_array, test=mask_psf(B, B.shape, (230,230))
        row_means = np.nanmedian(img_array, axis=1)
        B = B-row_means[:, np.newaxis] #-row_medians[:, np.newaxis]
        img_array1, test=mask_psf(B, B.shape, (230,230))
        column_medians = np.nanmedian(img_array1, axis=0)
        B = B-column_medians #-col_medians
        no_backgroundB.append(B)
        
        Light1=Science1[i]
        Light1=Light1*mask_inv
        Light1[Light1 == 0] = 'nan'
        Light1=Light1-Dark_Science
        Light1=Light1/Flat_norm
        mask = np.isnan(Light1)
        Light1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light1[~mask])
        A=(Light1[all_coorA[0]-dim:all_coorA[0]+dim,all_coorA[1]-dim:all_coorA[1]+dim]-Master_flat_B)
        img_array, test=mask_psf(A, A.shape, (230,230))
        row_means = np.nanmedian(img_array, axis=1)
        A= A-row_means[:, np.newaxis] #-row_medians[:, np.newaxis]
        img_array1, test=mask_psf(A, A.shape, (230,230))
        column_medians = np.nanmedian(img_array1, axis=0)
        A= A-column_medians #-col_medians
        no_backgroundA.append(A)
        
    no_backgroundA=np.array(no_backgroundA)
    mediaA=np.median(no_backgroundA, axis=0)
    coordinatesA = peak_local_max(mediaA[230-30:230+30,230-30:230+30], min_distance=5, threshold_abs=300)
    if len(coordinatesA) > 0:
        guessxA=coordinatesA[0][0]
        guessyA=coordinatesA[0][1]
    else:
        guessxA=30
        guessyA=30

    no_backgroundB=np.array(no_backgroundB)
    mediaB=np.median(no_backgroundB, axis=0)
    coordinatesB = peak_local_max(mediaB[230-30:230+30,230-30:230+30], min_distance=5, threshold_abs=300)
    if len(coordinatesB) > 0:
        guessxB=coordinatesB[0][0]
        guessyB=coordinatesB[0][1]
    else:
        guessxB=30
        guessyB=30

    fig0 = plt.figure()
    ax1 = plt.gca()
    z  = ZScaleInterval()
    z1, z2 = z.get_limits(mediaA[230-30:230+30,230-30:230+30])
    plt.imshow(mediaA[230-30:230+30,230-30:230+30], cmap=cm.hot, vmin = z1, vmax = z2)
    ax1.invert_yaxis()
    ax1.set_title('Leakage-term A position')
    ax1.set_xlabel('Pixel',fontsize=15)
    ax1.set_ylabel('Pixel',fontsize=15)
    plt.scatter(guessxA,guessyA)
    plt.grid(None)
    plt.colorbar(label='$counts$ $[a.u.]$')
    plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termA.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

    fig1 = plt.figure()
    ax1 = plt.gca()
    z  = ZScaleInterval()
    plt.imshow(mediaB[230-30:230+30,230-30:230+30], cmap=cm.hot, vmin = z1, vmax = z2)
    ax1.invert_yaxis()
    ax1.set_title('Leakage-term B position')
    ax1.set_xlabel('Pixel',fontsize=15)
    ax1.set_ylabel('Pixel',fontsize=15)
    plt.grid(None)
    plt.colorbar(label='$counts$ $[a.u.]$')
    plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termB.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

    for i in range(0,len(Science1)):
        #Pulisco righe e colonne
        A=no_backgroundA[i]
        immagine1=A[230-30:230+30,230-30:230+30]
        mask = np.isnan(immagine1)
        immagine1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), immagine1[~mask])
        x = np.linspace(0, 59, 60)
        y = np.linspace(0, 59, 60)
        X,Y = np.meshgrid(x,y)
        xdata = np.vstack((X.ravel(),Y.ravel()))
        immagine1_flat = immagine1.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine1_flat, p0=(500,guessxA,guessyA,8,8,0,0))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Position A File "+File_A[j])
            sys.exit(error_message)
        dx=popt[1]-30
        dy=popt[2]-30
        dxA.append(dx)
        dyA.append(dy)
        sigmaA.append(popt[3])
        shifted = imutils.translate(A, -dx, -dy)
        
        ScienceA.append(shifted)
    
        B=no_backgroundB[i]
        immagine=B[230-30:230+30,230-30:230+30]
        mask = np.isnan(immagine)
        immagine[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), immagine[~mask])
        immagine_flat = immagine.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(500,guessxB,guessxB,8,8,0,0))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Position B File "+File_B[j])
            sys.exit(error_message)
        dx1=popt[1]-30
        dy1=popt[2]-30
        dxB.append(dx1)
        dyB.append(dy1)
        sigmaB.append(popt[3])
        shifted = imutils.translate(B, -dx1, -dy1)
        
        ScienceB.append(shifted)
        
    ##Salvo i fits
    ScienceA=np.array(ScienceA)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_A[j]+'_diff.fits', ScienceA)
    ScienceB=np.array(ScienceB)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_B[j]+'_diff.fits', ScienceB)
    dxA=np.array(dxA)
    dyA=np.array(dyA)
    sigmaA=np.array(sigmaA)
    dxB=np.array(dxB)
    dyB=np.array(dyB)
    sigmaB=np.array(sigmaB)
    np.savetxt(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_A[j]+'info_diff.txt', np.transpose([np.linspace(0,len(Science1), len(Science1)),dxA,dyA,sigmaA]))
    np.savetxt(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_B[j]+'info_diff.txt', np.transpose([np.linspace(0,len(Science1), len(Science1)),dxB,dyB,sigmaB]))

    # File FITS di origine (da cui copiare l'header)
    file_origine = PATH+'raw/'+received_variable+'/object/'+File_A[j]+".fits"
    hdul_origine = fits.open(file_origine)
    
    # File FITS di destinazione (in cui copiare l'header)
    file_destinazione = PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_A[j]+"_diff.fits"
    hdul_destinazione = fits.open(file_destinazione, mode='update')  # Assicurati di aprire il file in modalità di aggiornamento ('update')
    
    header_origine = hdul_origine[0].header
    hdul_destinazione[0].header = header_origine
    
    hdul_destinazione.flush()
    hdul_destinazione.close()
    hdul_origine.close()

    # File FITS di origine (da cui copiare l'header)
    file_origine = PATH+'raw/'+received_variable+'/object/'+File_B[j]+".fits"
    hdul_origine = fits.open(file_origine)
    
    # File FITS di destinazione (in cui copiare l'header)
    file_destinazione = PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File_B[j]+"_diff.fits"
    hdul_destinazione = fits.open(file_destinazione, mode='update')  # Assicurati di aprire il file in modalità di aggiornamento ('update')
    
    header_origine = hdul_origine[0].header
    hdul_destinazione[0].header = header_origine
    
    hdul_destinazione.flush()
    hdul_destinazione.close()
    hdul_origine.close()