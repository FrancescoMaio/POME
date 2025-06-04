from matplotlib import pyplot as plt, cm
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import vip_hci as vip
import numpy as np
import os
import cv2
import pylab
import concurrent.futures

star    = 'as209'
path    = '/export/home/venere1/fedele/eris/'
offx    = 19.216 -1.25 
offy    = 0.965 +4.75 
nbin    = 100
nthread = 30  # Numero di thread da utilizzare
pclist  = [1,2,3,4,5,6,7,8,9,10,12,15,20]

def translate_image(image, dx, dy):
    # Ottenere le dimensioni dell'immagine
    height, width = image.shape[:2]

    # Definire la matrice di trasformazione per la traslazione
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])

    # Applicare la trasformazione all'immagine utilizzando la funzione warpAffine
    translated_image = cv2.warpAffine(image, M, (width, height))

    return translated_image

def ApriFits(percorso):
    image_file = get_pkg_data_filename(percorso)
    fits.info(image_file)
    immagine = fits.getdata(image_file, ext=0)
    return immagine

def header_reader_posang(percorso):
    hdul = fits.open(percorso)
    # Leggi il valore dall'header
    ang_start = hdul[0].header['ESO ADA POSANG']
    ang_end = hdul[0].header['ESO ADA POSANG END']
    # Chiudi il file FITS
    hdul.close()
    return ang_end,ang_start
    
def lista_file_fits(directory):
    # Ottieni la lista dei file nella directory
    lista_file = os.listdir(directory)

    # Filtra i file con estensione '.fits'
    file_fits = [file for file in lista_file if file.endswith('.fits')]

    return file_fits

def parallel_pca_processing(i):
    fr_pca1 = vip.psfsub.pca(cube, angle, ncomp=i, mask_center_px=None,
                             imlib=imlib, interpolation=interpolation,
                             svd_mode='lapack')
    # Salva il frame
    fits.writeto(path+'reduced/'+star+'/object/PCA/'+star+'_pc_'+str(i)+'.fits', fr_pca1)
    return fr_pca1

def calcola_media_batch(immagini, angoli_posizione, dimensione_batch):
    medie_batch_immagini = []
    medie_batch_angoli = []
    num_immagini = len(immagini)
    num_batch = num_immagini // dimensione_batch
    
    for i in range(num_batch):
        batch_immagini = immagini[i * dimensione_batch : (i + 1) * dimensione_batch]
        batch_angoli = angoli_posizione[i * dimensione_batch : (i + 1) * dimensione_batch]
        
        media_batch_immagini = np.mean(batch_immagini, axis=0)
        media_batch_angoli = np.mean(batch_angoli)
        
        medie_batch_immagini.append(media_batch_immagini)
        medie_batch_angoli.append(media_batch_angoli)
    
    if num_immagini % dimensione_batch != 0:
        resto_immagini = immagini[num_batch * dimensione_batch :]
        resto_angoli = angoli_posizione[num_batch * dimensione_batch :]
        
        media_restanti_immagini = np.mean(resto_immagini, axis=0)
        media_restanti_angoli = np.mean(resto_angoli)
        
        medie_batch_immagini.append(media_restanti_immagini)
        medie_batch_angoli.append(media_restanti_angoli)
    
    return medie_batch_immagini, medie_batch_angoli

File_target, File_sky = np.loadtxt(path+'raw/'+star+'/'+'list.txt', dtype=str, unpack=True)

cube=[]
angle=[]
for file in File_target:
    root=path+'/reduced/'+star+'/object/'+file
    immagini=[]
    angoli_posizione=[]
    img = ApriFits(root+'_object.fits')
    ang_end,ang_start = header_reader_posang(root+'_object.fits')
    x, var1, var2, xa, xb, xc, xd, xe, xf, xg, xh, x1a, xm, xn, xo = np.loadtxt(root+ '_info.txt', unpack=True)
    dx=np.median([xa,xe,x1a])+ offx #19.216 -1.25 # 19.169672225150634 +0.5
    dy=np.median([xb,xf,xm])+ offy #0.965 +4.75   # 0.7116814466855885 +5.75
    median = np.median(var1)
    MAD = np.median(np.abs(var1 - median))*5
    ang=ang_end-ang_start
    angoli=np.linspace(ang_start,ang_end,len(img))
    for index,i in enumerate(img):
        if var1[index] < median+MAD:
            i=translate_image(i.astype(np.float64),-dx,-dy)
            trimm=i[250-210:250+210,236-210:236+210]
            immagini.append(trimm)
            angoli_piccoli=(ang/len(img))*index
            angoli_posizione.append(angoli[index]+angoli_piccoli)
    medie_batch_immagini, medie_batch_angoli = calcola_media_batch(np.array(immagini), np.array(angoli_posizione), nbin)
    cube= cube+medie_batch_immagini
    angle= angle+medie_batch_angoli
    
print(np.array(cube).shape)
cube=np.array(cube).reshape(np.array(cube).shape[0], np.array(cube).shape[1], np.array(cube).shape[2])
print(np.array(cube).shape)
angle=np.array(angle).reshape(np.array(cube).shape[0],)
print(np.array(angle).shape)


imlib = 'vip-fft'
interpolation=None

# Impostaziomniu

# args = list(range(1, 20))
with concurrent.futures.ThreadPoolExecutor(max_workers=nthread) as executor:
    # Mappiamo la funzione su ogni argomento in parallelo
    risultati = list(executor.map(parallel_pca_processing, pclist))
