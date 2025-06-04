'''
In queesto programma aapro tutti i file Dark e li medio, qusto permette di creare il Master_Dark. Da questo poi estraiamo l'istogramma, lo fittiamo con una gaussiane tutto quello che è sopra e sotto di 5 sigma dalla meda lo marchiamo come bad pixel e lo andiamo a creare una mappa di qiesti punti.
'''

from matplotlib import pyplot as plt, cm
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from matplotlib import colors
from scipy.stats import norm
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import glob
import cv2
import sys
import os
from astropy.visualization import ZScaleInterval

if len(sys.argv) != 3:
    print("Usage: python eris_calib_dark_psf_w4.py <variable1> <variable2>")
    sys.exit(1)

PATH = sys.argv[1]
received_variable = sys.argv[2]

print(f"Script received the variables: {PATH}, {received_variable}")

def ApriFits(percorso):
    image_file = get_pkg_data_filename(percorso)
    fits.info(image_file)
    immagine = fits.getdata(image_file, ext=0)
    return immagine

##Apro dark
dark = []
print(PATH+"raw/"+received_variable+"/calib/dark_psf/*.fits")
files = glob.glob (PATH+"raw/"+received_variable+"/calib/dark_psf/*.fits") 
for myFile in files:
    immagine = ApriFits(myFile)
    dark.append(immagine)
dark=np.array(dark)
Master_Dark=np.mean(dark, axis=0)
# Create a directory
folder_name = "master_dark"
os.makedirs(PATH+'reduced/'+received_variable+'/calib/dark_psf/'+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")
fits.writeto(PATH+'reduced/'+received_variable+'/calib/dark_psf/master_dark/master_dark.fits', Master_Dark)


