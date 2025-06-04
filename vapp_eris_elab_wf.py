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
from astropy.visualization import ZScaleInterval
import os
import sys
import logging

# Set up logging
log_file = "fit_errors.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s - %(message)s")

if len(sys.argv) != 3:
    print("Usage: python vapp_eris_elab_wf.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

##Definizione delle costanti
dim=350
center_down = (200, 463)
center_up = (500, 237)
all_coorB=(1510,1471)
all_coorA=(1528,788)

radius = 120

def mask_psf(image, shape, coor):
    center_down = (200, 463)
    center_up = (500, 237)
    mask_1 = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask_1, coor, 40, 255, thickness=-1)
    cv2.circle(mask_1, (235,200), 20, 255, thickness=-1)
    cv2.circle(mask_1, (465,500), 20, 255, thickness=-1)
    cv2.circle(mask_1, (133,532), 50, 255, thickness=-1)
    cv2.circle(mask_1, (560,166), 50, 255, thickness=-1)
    cv2.circle(mask_1, (430,128), 30, 255, thickness=-1)
    cv2.circle(mask_1, (264,562), 30, 255, thickness=-1)
    cv2.circle(mask_1, (132,347), 30, 255, thickness=-1)
    cv2.circle(mask_1, (571,350), 40, 255, thickness=-1)
    cv2.circle(mask_1, (300,430), 20, 255, thickness=-1)
    cv2.circle(mask_1, (400,260), 40, 255, thickness=-1)
    cv2.circle(mask_1, (300,430), 40, 255, thickness=-1)
    cv2.circle(mask_1, (287,522), 20, 255, thickness=-1) 
    cv2.circle(mask_1, (638,110), 20, 255, thickness=-1)
    cv2.circle(mask_1, (287,522), 40, 255, thickness=-1)     
    cv2.circle(mask_1, (404,181), 20, 255, thickness=-1)
    cv2.circle(mask_1, (312,536), 20, 255, thickness=-1)   
    cv2.circle(mask_1, (198,585), 50, 255, thickness=-1)
    cv2.circle(mask_1, (490,120), 40, 255, thickness=-1) 
    cv2.circle(mask_1, (367,444), 40, 255, thickness=-1)
    cv2.circle(mask_1, (325,260), 50, 255, thickness=-1)  
    cv2.circle(mask_1, (40,610), 30, 255, thickness=-1)
    cv2.circle(mask_1, (665,73), 30, 255, thickness=-1)
    cv2.circle(mask_1, (305,381), 30, 255, thickness=-1)
    cv2.circle(mask_1, (384,308), 30, 255, thickness=-1)
    cv2.circle(mask_1, (284,592), 20, 255, thickness=-1)
    cv2.circle(mask_1, (384,197), 40, 255, thickness=-1)
    cv2.circle(mask_1, (319,560), 30, 255, thickness=-1)
    cv2.circle(mask_1, (111,319), 30, 255, thickness=-1)
    cv2.circle(mask_1, (353,400), 30, 255, thickness=-1)
    cv2.circle(mask_1, (132,581), 20, 255, thickness=-1)
    
    #cv2.circle(mask_1, (431,336), 30, 255, thickness=-1)  #Solo per HD169142
    
    cv2.circle(mask_1, center_down, 100, 255, thickness=-1)
    cv2.circle(mask_1, center_up, 100, 255, thickness=-1)
    mask_inv = mask_1
    mask_inv[mask_inv==0] = 1
    mask_inv[mask_inv==255] = 0

    masked_image=image*mask_inv
    masked_image[masked_image == 0] = 'nan'
    return masked_image, mask_1

def draw_half_circle(ax, center, radius, direction):
    angle_start, angle_end = (0, 180) if direction == 'down' else (180, 360)
    angle_start_rad = np.radians(-angle_start - 40)  # Ruota di -40° rispetto all'inclinazione richiesta
    angle_end_rad = np.radians(-angle_end - 40)
    
    if direction == 'down':
        arc = Arc(center, 2*radius, 2*radius, angle=-180, theta1=np.degrees(angle_start_rad), theta2=np.degrees(angle_end_rad), edgecolor='blue', fill=False)
    else:
        arc = Arc(center, 2*radius, 2*radius, angle=-180, theta1=np.degrees(angle_start_rad), theta2=np.degrees(angle_end_rad), edgecolor='blue', fill=False)
    ax.add_patch(arc)

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

# Check if arrays have the same length
if len(File_A) != len(File_B):
    error_message = f"Error: Mismatch in number of files. File_A={len(File_A)}, File_B={len(File_B)}"
    print(error_message)
    logging.error(error_message)
    sys.exit(1)  # Exit the program with error code 1

folder_name = "clean"
os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")

for i in tqdm(range(0,len(File_A)), desc='Background remuval'):

    all_coorB=(xb[i],yb[i])
    all_coorA=(xa[i],ya[i])
    ##Master_flat_A
    Master_flat_B = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+FileB[i]+'Master_flatB.fits')
    
    ##Master_flat_B
    Master_flat_A = ApriFits(PATH+'reduced/'+received_variable+'/sky/'+FileA[i]+'Master_flatA.fits')
    
    ##Science
    Science = ApriFits(PATH+'raw/'+received_variable+'/object/'+FileA[i]+'.fits')
    
    Flat_s=Science[1]
    Flat_s=Flat_s*mask_inv
    Flat_s[Flat_s == 0] = 'nan'
    Flat_s=Flat_s-Dark_Science
    
    lim_inf=270
    lim_sup=1910
    
    ##Science1
    Science1 = ApriFits(PATH+'raw/'+received_variable+'/object/'+FileB[i]+'.fits')
    ScienceA = []
    dxA = []
    dyA = []
    sigmaA = []
    ScienceB = []
    dxB = []
    dyB = []
    sigmaB = []
    
    for i in range(0,len(Science1)):
        Light=Science[i]
        Light=Light*mask_inv
        Light[Light == 0] = 'nan'
        Light=Light-Dark_Science
        Light=Light/Flat_norm
        mask = np.isnan(Light)
        Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
    
        
        Light1=Science1[i]
        Light1=Light1*mask_inv
        Light1[Light1 == 0] = 'nan'
        Light1=Light1-Dark_Science
        Light1=Light1/Flat_norm
        mask = np.isnan(Light1)
        Light1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light1[~mask])
    
        #Faccio un fit gaussiano per trovare il picco centrale
        initial_guess = (17000,30,30,5,5,0,0) #14700
        A=(Light1[all_coorA[0]-dim:all_coorA[0]+dim,all_coorA[1]-dim:all_coorA[1]+dim]-Master_flat_B)
        img_array, test=mask_psf(A, A.shape, (350,350))
        row_means = np.nanmedian(img_array, axis=1)
        A= A-row_means[:, np.newaxis] #-row_medians[:, np.newaxis]
        img_array1, test=mask_psf(A, A.shape, (350,350))
        column_medians = np.nanmedian(img_array1, axis=0)
        A= A-column_medians #-col_medians
        
        immagine1=Light1[all_coorA[0]-30:all_coorA[0]+30,all_coorA[1]-30:all_coorA[1]+30]
        mask = np.isnan(immagine1)
        immagine1[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), immagine1[~mask])
        
        if i==0:
            fig0 = plt.figure()
            ax1 = plt.gca()
            z  = ZScaleInterval()
            z1, z2 = z.get_limits(immagine1)
            plt.imshow(immagine1, cmap=cm.hot, vmin = z1, vmax = z2)
            ax1.invert_yaxis()
            ax1.set_title('Leakage-term A position')
            ax1.set_xlabel('Pixel',fontsize=15)
            ax1.set_ylabel('Pixel',fontsize=15)
            plt.grid(None)
            plt.colorbar(label='$counts$ $[a.u.]$')
            plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termA.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine
        x = np.linspace(0, 59, 60)
        y = np.linspace(0, 59, 60)
        X,Y = np.meshgrid(x,y)
        xdata = np.vstack((X.ravel(),Y.ravel()))
        immagine1_flat = immagine1.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(10000,30,30,5,5,0,17500))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Position A File "+FileA[i])
            sys.exit(error_message)
        dx=popt[1]-30 
        dy=popt[2]-30
        dxA.append(dx)
        dyA.append(dy)
        sigmaA.append(popt[3])
        shifted = imutils.translate(A, -dx, -dy)
        
        ScienceA.append(A)
    
        
        B=(Light[all_coorB[0]-dim:all_coorB[0]+dim,all_coorB[1]-dim:all_coorB[1]+dim]-Master_flat_A)
        img_array, test=mask_psf(B, B.shape, (350,350))
        row_means = np.nanmedian(img_array, axis=1)
        B = B-row_means[:, np.newaxis] #-row_medians[:, np.newaxis]
        img_array1, test=mask_psf(B, B.shape, (350,350))
        column_medians = np.nanmedian(img_array1, axis=0)
        B = B-column_medians #-col_medians
        
        immagine=Light[all_coorB[0]-30:all_coorB[0]+30,all_coorB[1]-30:all_coorB[1]+30]
        mask = np.isnan(immagine)
        immagine[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), immagine[~mask])
        if i==0:
            fig1 = plt.figure()
            ax1 = plt.gca()
            z  = ZScaleInterval()
            z1, z2 = z.get_limits(immagine)
            plt.imshow(immagine, cmap=cm.hot, vmin = z1, vmax = z2)
            ax1.invert_yaxis()
            ax1.set_title('Leakage-term B position')
            ax1.set_xlabel('Pixel',fontsize=15)
            ax1.set_ylabel('Pixel',fontsize=15)
            plt.grid(None)
            plt.colorbar(label='$counts$ $[a.u.]$')
            plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termB.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine
        immagine_flat = immagine.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(10000,30,30,5,5,0,17500))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Position B File "+FileB[i])
            sys.exit(error_message)
        dx1=popt[1]-30
        dy1=popt[2]-30
        dxB.append(dx1)
        dyB.append(dy1)
        sigmaB.append(popt[3])
        shifted = imutils.translate(B, -dx1, -dy1)
        
        ScienceB.append(B)
        
    ##Salvo i fits
    ScienceA=np.array(ScienceA)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File1[i]+'_diff.fits', ScienceA)
    ScienceB=np.array(ScienceB)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File2[i]+'_diff.fits', ScienceB)
    dxA=np.array(dxA)
    dyA=np.array(dyA)
    sigmaA=np.array(sigmaA)
    dxB=np.array(dxB)
    dyB=np.array(dyB)
    sigmaB=np.array(sigmaB)
    np.savetxt(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File1[i]+'info_diff.txt', np.transpose([np.linspace(0,len(Science1), len(Science1)),dxA,dyA,sigmaA]))
    np.savetxt(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File2[i]+'info_diff.txt', np.transpose([np.linspace(0,len(Science1), len(Science1)),dxB,dyB,sigmaB]))
    
    ##Plot
    fig2 = plt.figure()
    ax1 = plt.gca()
    z  = ZScaleInterval()
    z1, z2 = z.get_limits(np.median(ScienceA, axis=0))
    plt.imshow(np.median(ScienceA, axis=0), cmap=cm.hot, vmin = z1, vmax = z2)
    ax1.invert_yaxis()
    ax1.set_title('vAPP PSF ScienceA')
    ax1.set_xlabel('Pixel',fontsize=15)
    ax1.set_ylabel('Pixel',fontsize=15)
    plt.grid(None)
    plt.colorbar(label='$counts$ $[a.u.]$')
    plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termB.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

    fig3 = plt.figure()
    ax1 = plt.gca()
    z  = ZScaleInterval()
    z1, z2 = z.get_limits(np.median(ScienceB, axis=0))
    plt.imshow(np.median(ScienceB, axis=0), cmap=cm.hot, vmin = z1, vmax = z2)
    ax1.invert_yaxis()
    ax1.set_title('vAPP PSF ScienceB')
    ax1.set_xlabel('Pixel',fontsize=15)
    ax1.set_ylabel('Pixel',fontsize=15)
    plt.grid(None)
    plt.colorbar(label='$counts$ $[a.u.]$')
    plt.savefig(PATH+"reduced/"+received_variable+"/image/leakage_termB.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine
