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
import logging

# Set up logging
log_file = "fit_errors.log"
logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s - %(message)s")

if len(sys.argv) != 3:
    print("Usage: python eris_elab_Obj_w3.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

spot1=(50,1150)
spot2=(161,690)
spot3=(230,1255)
spot4=(300,1369)
spot5=(403,935)
spot6=(429,1286)

##Definizione delle costanti
dim=256
coor = (257, 1045)

def translate_image(image, dx, dy):
    #print(dx,dy)
    # Ottenere le dimensioni dell'immagine
    height, width = image.shape[:2]

    # Definire la matrice di trasformazione per la traslazione
    M = np.float32([[1, 0, dx],
                   [0, 1, dy]])

    # Applicare la trasformazione all'immagine utilizzando la funzione warpAffine
    translated_image = cv2.warpAffine(image, M, (width, height))

    return translated_image

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

def mask_psf(image, shape, coor):
    mask_1 = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask_1, coor, 100, 255, thickness=-1)
    
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

def find_max_position_in_subframe(frame, coor, box_size=50):
    """
    Find the position of the maximum value within a 20x20 subframe centered at the specified coordinates.
    
    Parameters:
        frame (2D np.ndarray): The image/frame in which to search.
        coor (tuple): Center coordinates (y, x).
        box_size (int): Size of the square subframe (default is 20).
    
    Returns:
        max_pos (tuple): Global coordinates (y, x) of the maximum value in the original frame.
    """
    y, x = coor
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
    
##Apro Object e Dark

File_target = np.loadtxt(PATH+'raw/'+received_variable+'/'+'obj_list.txt', dtype=str, unpack=True)
File_sky = np.loadtxt(PATH+'raw/'+received_variable+'/'+'sky_list.txt', dtype=str, unpack=True)

# Check if arrays have the same length
if len(File_sky) != len(File_target):
    error_message = f"Error: Mismatch in number of files. File_sky={len(File_sky)}, File_target={len(File_target)}"
    print(error_message)
    logging.error(error_message)
    sys.exit(1)  # Exit the program with error code 1


window=(,)

##Apro flat    
flat = ApriFits(PATH+'reduced/'+received_variable+'/calib/flat/master_flat/master_flat.fits')
Flat=np.array(flat)
Flat=Flat[1154:1154+512,0:2048]

##Apro Science dark
dark_sc = ApriFits(PATH+'reduced/'+received_variable+'/calib/dark_obj/master_dark/master_dark.fits')
Master_Dark_Science=np.array(dark_sc)

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

for n,File in enumerate(File_target):
    ##Science
    Science = ApriFits(PATH+'raw/'+received_variable+'/object/'+File+'.fits')
    File_sky_n=File_sky[n]
    Sky = ApriFits(PATH+'reduced/'+received_variable+'/sky/media/'+File_sky_n+'_sky.fits')
    if n ==0:
        spot1=find_max_position_in_subframe(Sky, spot1)
        spot2=find_max_position_in_subframe(Sky, spot2)
        spot3=find_max_position_in_subframe(Sky, spot3)
        spot4=find_max_position_in_subframe(Sky, spot4)
        spot5=find_max_position_in_subframe(Sky, spot5)
        spot6=find_max_position_in_subframe(Sky, spot6)
    spot1S=Sky[spot1[1]-30:spot1[1]+30,spot1[0]-30:spot1[0]+30]
    spot2S=Sky[spot2[1]-30:spot2[1]+30,spot2[0]-30:spot2[0]+30]
    spot3S=Sky[spot3[1]-30:spot3[1]+30,spot3[0]-30:spot3[0]+30]
    spot4S=Sky[spot4[1]-30:spot4[1]+30,spot4[0]-30:spot4[0]+30]
    spot5S=Sky[spot5[1]-30:spot5[1]+30,spot5[0]-30:spot5[0]+30]
    spot6S=Sky[spot6[1]-30:spot6[1]+30,spot6[0]-30:spot6[0]+30]
    ScienceA = []
    print(File)
    print(File_sky_n)
    Dx1L = []
    Dy1L = []
    Dx1S = []
    Dy1S = []
    Dx2L = []
    Dy2L = []
    Dx2S = []
    Dy2S = []
    Dx3L = []
    Dy3L = []
    Dx3S = []
    Dy3S = []
    Dx4L = []
    Dy4L = []
    Dx4S = []
    Dy4S = []
    Dx5L = []
    Dy5L = []
    Dx5S = []
    Dy5S = []
    Dx6L = []
    Dy6L = []
    Dx6S = []
    Dy6S = []
    
    for i in tqdm(range(0,len(Science)), desc='Background remuval'):
        Light=Science[i]
        Light=Light*mask_inv
        Light[Light == 0] = 'nan'
        Light=Light-Dark_Science
        Light=Light/Flat_norm
        mask = np.isnan(Light)
        Light[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Light[~mask])
        spot1L=Light[spot1[1]-30:spot1[1]+30,spot1[0]-30:spot1[0]+30]
        spot2L=Light[spot2[1]-30:spot2[1]+30,spot2[0]-30:spot2[0]+30]
        spot3L=Light[spot3[1]-30:spot3[1]+30,spot3[0]-30:spot3[0]+30]
        spot4L=Light[spot4[1]-30:spot4[1]+30,spot4[0]-30:spot4[0]+30]
        spot5L=Light[spot5[1]-30:spot5[1]+30,spot5[0]-30:spot5[0]+30]
        spot6L=Light[spot6[1]-30:spot6[1]+30,spot6[0]-30:spot6[0]+30]
        Light=Light[coor[0]-dim:coor[0]+dim,coor[1]-dim:coor[1]+dim]
        
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
        Dx1L.append(dx1L)
        Dy1L.append(dy1L)
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
        Dx1S.append(dx1S)
        Dy1S.append(dy1S)
       
        immagine_flat = spot2L.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 2L")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan) 
        dx2L=popt[1]-30
        dy2L=popt[2]-30
        Dx2L.append(dx2L)
        Dy2L.append(dy2L)
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
        Dx2S.append(dx2S)
        Dy2S.append(dy2S)
        
        mask = np.isnan(spot3L)
        spot3L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot3L[~mask])
        immagine_flat = spot3L.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            print(error_message)
            logging.error(error_message+" Spot 3L")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx3L=popt[1]-30
        dy3L=popt[2]-30
        Dx3L.append(dx3L)
        Dy3L.append(dy3L)
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
        Dx3S.append(dx3S)
        Dy3S.append(dy3S)
        
        mask = np.isnan(spot4L)
        spot4L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot4L[~mask])
        immagine_flat = spot4L.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 4L")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx4L=popt[1]-30
        dy4L=popt[2]-30
        Dx4L.append(dx4L)
        Dy4L.append(dy4L)
        mask = np.isnan(spot4S)
        spot4S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot4S[~mask])
        immagine_flat = spot4S.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 4S")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx4S=popt[1]-30
        dy4S=popt[2]-30
        Dx4S.append(dx4S)
        Dy4S.append(dy4S)
        
        mask = np.isnan(spot5L)
        spot5L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot5L[~mask])
        immagine_flat = spot5L.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 5L")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx5L=popt[1]-30
        dy5L=popt[2]-30
        Dx5L.append(dx5L)
        Dy5L.append(dy5L)
        mask = np.isnan(spot5S)
        spot5S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot5S[~mask])
        immagine_flat = spot5S.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 5S")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx5S=popt[1]-30
        dy5S=popt[2]-30
        Dx5S.append(dx5S)
        Dy5S.append(dy5S)
        
        mask = np.isnan(spot6L)
        spot6L[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot6L[~mask])
        immagine_flat = spot6L.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 6L")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx6L=popt[1]-30
        dy6L=popt[2]-30
        Dx6L.append(dx6L)
        Dy6L.append(dy6L)
        mask = np.isnan(spot6S)
        spot6S[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), spot6S[~mask])
        immagine_flat = spot6S.flatten()
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, xdata, immagine_flat, p0=(2500,30,30,5,5,0,1800))
        except (opt.OptimizeWarning, RuntimeError) as e:
            error_message = f"Fit did not converge: {str(e)}"
            logging.error(error_message+" Spot 6S")
        
            # Assign NaN values on failure
            popt = np.full(7, np.nan) 
            pcov = np.full((7, 7), np.nan)
        dx6S=popt[1]-30
        dy6S=popt[2]-30
        Dx6S.append(dx6S)
        Dy6S.append(dy6S)
        
        medianxS=np.nanmedian([dx1S,dx2S,dx3S,dx4S,dx5S,dx6S])
        medianyS=np.nanmedian([dy1S,dy2S,dy3S,dy4S,dy5S,dx6S])
        medianxL=np.nanmedian([dx1L,dx2L,dx3L,dx4L,dx5L,dx6S])
        medianyL=np.nanmedian([dy1L,dy2L,dy3L,dy4L,dy5L,dx6S])

        offx=medianxL-medianxS
        offy=medianyL-medianyS

        Sky1 = translate_image(Sky.astype(np.float64), offx, offy)
        
        Light=Light-Sky1[coor[0]-dim:coor[0]+dim,coor[1]-dim:coor[1]+dim]
    
        img_array, test=mask_psf(Light, Light.shape, (dim,dim))
        row_means = np.nanmedian(img_array, axis=1)
        Light= Light-row_means[:, np.newaxis]
        img_array1, test=mask_psf(Light, Light.shape, (dim,dim))
        column_medians = np.nanmedian(img_array1, axis=0)
        Light= Light-column_medians
        
        ScienceA.append(Light)

    
    ##Salvo i fits
    folder_name = "object"
    os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)
    print(f"Directory '{folder_name}' created (or already exists).")
    ScienceA=np.array(ScienceA)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+"_object.fits", ScienceA)
    median=np.median(ScienceA, axis=0)
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+'_test.fits', median)
    difference=ScienceA-median
    del Science

    valori_medi = []
    sigma = []
    for x,i in tqdm(enumerate(difference), desc='Stima centraggio'):
        valore_medio = np.mean(i)
        valori_medi.append(valore_medio)
        std = np.std(i)
        sigma.append(std)
    valori_medi=np.array(valori_medi)
    sigma=np.array(sigma)
    np.savetxt(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+'_info.txt', np.transpose([np.linspace(1,len(ScienceA), len(ScienceA)),valori_medi,sigma,np.array(Dx1S),np.array(Dy1S),np.array(Dx1L),np.array(Dy1L),np.array(Dx2S),np.array(Dy2S),np.array(Dx2L),np.array(Dy2L),np.array(Dx3S),np.array(Dy3S),np.array(Dx3L),np.array(Dy3L)]))
    
    # File FITS di origine (da cui copiare l'header)
    file_origine = PATH+'raw/'+received_variable+'/object/'+File+".fits"
    hdul_origine = fits.open(file_origine)
    
    # File FITS di destinazione (in cui copiare l'header)
    file_destinazione = PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+"_object.fits"
    hdul_destinazione = fits.open(file_destinazione, mode='update') 
    
    header_origine = hdul_origine[0].header
    hdul_destinazione[0].header = header_origine
    
    hdul_destinazione.flush()
    hdul_destinazione.close()
    hdul_origine.close()
    del ScienceA


##Plot
fig2 = plt.figure()
ax1 = plt.gca()
z  = ZScaleInterval()
z1, z2 = z.get_limits(median)
plt.imshow(median, cmap=cm.hot, vmin = z1, vmax = z2)
ax1.invert_yaxis()
ax1.set_title('Example median')
ax1.set_xlabel('Pixel',fontsize=15)
ax1.set_ylabel('Pixel',fontsize=15)
plt.grid(None)
plt.colorbar(label='$counts$ $[a.u.]$')

plt.savefig(PATH+"reduced/"+received_variable+"/image/example_median.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine


