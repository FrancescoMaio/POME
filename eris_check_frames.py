from matplotlib import pyplot as plt, cm
import pylab
import numpy as np
import os

targetname='as209'
path='/export/home/venere1/fedele/eris/'


File_target, File_sky = np.loadtxt(path+'raw/'+targetname+'/'+'list.txt', dtype=str, unpack=True)

for n,File in enumerate(File_target):
    print(File)
    x, var1, var2, xa, xb, xc, xd, xe, xf, xg, xh, x1a, xm, xn, xo = pylab.loadtxt(path+'reduced/'+targetname+'/object/'+File+'_info.txt', unpack = True)

    # Crea il secondo plot per le altre due variabili in funzione di x
    plt.figure(2)
    plt.plot(x, var1, label=n)
    plt.xlabel('DIT')
    plt.ylabel('Arbitray Unit')
    plt.title('Image Luminosity')
    plt.legend()

# Mostra i grafici
#plt.show()
plt.savefig('frame_flux_as209', dpi=300)
