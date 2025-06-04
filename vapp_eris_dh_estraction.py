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
import imutils
import numpy as np
import glob
import cv2
import os
import sys
from astropy.visualization import ZScaleInterval

if len(sys.argv) != 3:
    print("Usage: python vapp_eris_dh_estraction.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

# Define the center and radius of the half-moon shape (you may need to adjust these values)
center = (502, 238) # (x, y) coordinates of the center of the first half-moon
center1 = (199, 461)  # (x, y) coordinates of the center of the second half-moon
radius = 120 #200

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

def draw_half_circle(ax, center, radius, direction):
    angle_start, angle_end = (0, 180) if direction == 'down' else (180, 360)
    angle_start_rad = np.radians(-angle_start - 36)  # Ruota di -40° rispetto all'inclinazione richiesta
    angle_end_rad = np.radians(-angle_end - 36)
    
    if direction == 'down':
        arc = Arc(center, 2*radius, 2*radius, angle=-180, theta1=np.degrees(angle_start_rad), theta2=np.degrees(angle_end_rad), edgecolor='blue', fill=False)
    else:
        arc = Arc(center, 2*radius, 2*radius, angle=-180, theta1=np.degrees(angle_start_rad), theta2=np.degrees(angle_end_rad), edgecolor='blue', fill=False)
    ax.add_patch(arc)


def ApriFits(percorso):
    image_file = get_pkg_data_filename(percorso)
    fits.info(image_file)
    immagine = fits.getdata(image_file, ext=0)
    return immagine

def create_half_moon_mask(image_shape, center, radius, direction):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    start_angle, end_angle = (180, 360) if direction == 'up' else (0, 180)
    cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
    return mask

def rotate_image(image, angle, center):
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))

def translate_image(image, translation):
    rows, cols = image.shape[:2]
    translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    return cv2.warpAffine(image, translation_matrix, (cols, rows))


File_A, File_B, xa,ya,xb,yb = np.loadtxt('list.txt', dtype=str, unpack=True)
lista=np.concatenate((File_A,File_B))

folder_name = "object"
os.makedirs(PATH+"reduced/"+received_variable+"/"+folder_name, exist_ok=True)

##Apro fit 
for File in tqdm(lista, desc='Darh hole estraction'):

    B = ApriFits(PATH+"reduced/"+received_variable+"/clean/"+File+'_diff.fits')
    
    ScienceB = []
    for i in range(0,len(B)):
        # Load the image
        image = B[i]
        image=image.astype(np.float64)
        
        # Choose the direction of the half-moon ('up' or 'down')
        direction = 'down'
        
        # Create the half-moon mask
        half_moon_mask = create_half_moon_mask(image.shape, (350,351), radius, direction)
        
        # Rotate the image mask by -36 degrees [Manual p113.1]
        image=rotate_image(image,-36,(350,350))
        
        #traslate the image to center the down psf
        image_trasl = imutils.translate(image, -188.20342009389094, 0)#3.37) 192.2138703296697
        
        # Extract the half-moon region from the image using the rotated mask
        extracted_half_moon = cv2.bitwise_and(image_trasl, image_trasl, mask=half_moon_mask)

        # Choose the direction of the half-moon ('up' or 'down')
        direction = 'up'
        
        # Create the half-moon mask
        half_moon_mask = create_half_moon_mask(image.shape, (350,350), radius, direction)
        
        #traslate the image to center the down psf7
        image1_trasl = imutils.translate(image, 188.1937818311757, 0)#-3.4399999999999977) 192.07476814896035
        
        # Extract the half-moon region from the image using the rotated mask
        extracted_half_moon1 = cv2.bitwise_and(image1_trasl, image1_trasl, mask=half_moon_mask)
    
        #Sum the half-moons
        totalimage = extracted_half_moon1+ extracted_half_moon
        
        #Cut image to 400x400
        totalimage = totalimage[totalimage.shape[0]//2-150:totalimage.shape[0]//2+150,totalimage.shape[1]//2-150:totalimage.shape[1]//2+150]
        
        # Rotate the image mask by 36 degrees [Manual p113.1]
        totalimage=rotate_image(totalimage,36,(150,150))
        ScienceB.append(totalimage)
        
    ScienceB=np.array(ScienceB)
    
    fits.writeto(PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+'_cut.fits', ScienceB)
    
    # File FITS di origine (da cui copiare l'header)
    file_origine = PATH+'raw/'+received_variable+'/object/'+File+".fits"
    hdul_origine = fits.open(file_origine)
    
    # File FITS di destinazione (in cui copiare l'header)
    file_destinazione = PATH+"reduced/"+received_variable+"/"+folder_name+"/"+File+"_cut.fits"
    hdul_destinazione = fits.open(file_destinazione, mode='update')  # Assicurati di aprire il file in modalità di aggiornamento ('update')
    
    header_origine = hdul_origine[0].header
    hdul_destinazione[0].header = header_origine
    
    hdul_destinazione.flush()
    hdul_destinazione.close()
    hdul_origine.close()

##Plot quadrato
z  = ZScaleInterval()
fig2 = plt.figure()
plt.grid(None)
z1, z2 = z.get_limits(totalimage)
plt.imshow(totalimage,cmap=cm.hot, vmin = z1, vmax = z2)
plt.title('Extracted dark hole')

plt.savefig(PATH+"reduced/"+received_variable+"/image/dark_hole.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine
