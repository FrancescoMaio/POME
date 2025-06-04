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
    print("Usage: python eris_calib_dark_flat_w4.py <variable1> <variable2>")
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
files = glob.glob (PATH+"raw/"+received_variable+"/calib/dark_obj/*.fits") 
for myFile in files:
    immagine = ApriFits(myFile)
    dark.append(immagine)
dark=np.array(dark)
Master_Dark=np.mean(dark, axis=0)
print(dark.shape)
# Create a directory
folder_name = "master_dark"
os.makedirs(PATH+'reduced/'+received_variable+'/calib/dark_obj/'+folder_name, exist_ok=True)
print(f"Directory '{folder_name}' created (or already exists).")
fits.writeto(PATH+'reduced/'+received_variable+'/calib/dark_obj/master_dark/master_dark.fits', Master_Dark)

##Plot image
fig = plt.figure()
ax1 = plt.gca()
plt.imshow(Master_Dark, cmap=cm.hot)
ax1.invert_yaxis()
ax1.set_title('Mean of the dark frames')
ax1.set_xlabel('Pixel',fontsize=15)
ax1.set_ylabel('Pixel',fontsize=15)
plt.grid(None)
plt.colorbar(label='$counts$ $[a.u.]$')

plt.savefig(PATH+"reduced/"+received_variable+"/image/master_dark.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

##Instogramma per selezionare i bad pixel
k= 5 #Sgma del k-clipping
fig2 = plt.figure()
hinst=Master_Dark.ravel()
max=max(hinst)
n, bins, patches = plt.hist(hinst, 100, density=1)
hinst1 = hinst[(hinst < 25000)]
(mu, sigma) = norm.fit(hinst1)
soglia=mu+k*sigma
sogliad=mu-k*sigma
plt.axvline(x=soglia, color='red')
plt.axvline(x=sogliad, color='red')
plt.title('Dark Histogram',fontsize=25)
plt.xlabel('Counts [a.u.]',fontsize=20)
plt.ylabel('Occurrences',fontsize=20)
y = norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)

# Calcola il riquadro di zoom
zoom_data = hinst[(hinst >= 28000) & (hinst <= 42000)]

# Aggiungi un riquadro con uno zoom nella parte superiore a destra
left, bottom, width, height = 0.55, 0.5, 0.3, 0.3
ax_zoom = plt.axes([left, bottom, width, height])
ax_zoom.hist(zoom_data, bins=100, density=1)

# Aggiungi un rettangolo di zoom
#rect = Rectangle((30000, 0), (45000 - 30000), ax_zoom.get_ylim()[1], linewidth=1, edgecolor='black', facecolor='none')
#ax_zoom.add_patch(rect)

# Imposta i limiti del rettangolo di zoom
ax_zoom.set_xlim([30000, 45000])
ax_zoom.set_title('Bad Pixel Histogram',fontsize=15)
ax_zoom.set_xlabel('Counts [a.u.]',fontsize=10)
ax_zoom.set_ylabel('Occurrences',fontsize=10)
ax_zoom.set_ylim([0, 0.0002])  # Imposta i limiti sull'asse y del riquadro di zoom

plt.savefig(PATH+"reduced/"+received_variable+"/image/dark_instogram.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine

##Maschera
Master_Dark[Master_Dark<sogliad]= soglia+10 # In caso ci siano valori negativi li porta sopra soglia
mask = cv2.threshold(Master_Dark, soglia, max, cv2.THRESH_BINARY)[1]/max
mask_inv = mask
mask_inv[mask_inv==1] = 10
mask_inv[mask_inv==0] = 1
mask_inv[mask_inv==10] = 0

fits.writeto(PATH+'reduced/'+received_variable+'/calib/bad_pixel_mask.fits', mask_inv)

fig3 = plt.figure()
plt.imshow(mask_inv)
ax = plt.gca()
ax.invert_yaxis()
ax.set_title('Bad pixel mask')
ax.set_xlabel('Pixel',fontsize=15)
ax.set_ylabel('Pixel',fontsize=15)
plt.grid(None)

plt.savefig(PATH+"reduced/"+received_variable+"/image/bad_pixel_mask.png", bbox_inches='tight', pad_inches=0)  # Salva il grafico come immagine
