import numpy as np
from IOFunctions import *

#load the images
img04 = readNumpy('C:/DTEProjektpraktikumSebastianHeid/Vergleichsbilder/4_1024x1024.raw')
img26 = readNumpy('C:/DTEProjektpraktikumSebastianHeid/Vergleichsbilder/26_1024x1024.raw')
patient = readNumpy('C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S163/z-49_y10_x-60_pPrim_1024x1024x72.raw')

I0 = 2000

# make sure that no entries in img04 and img26 are zero

for i in range(0,1024):
    for j in range(0,1024):
        if(img04[i][j] == 0):
            img04[i][j] = I0

for i in range(0,1024):
    for j in range(0,1024):
        if(img26[i][j] == 0):
            img26[i][j] = I0

# calculate the p-value
im04 = -np.log(img04/I0)
im26 = -np.log(img26/I0)


writeNumpy(im04, 'C:/DTEProjektpraktikumSebastianHeid/Vergleichsbilder/img4.raw')
writeNumpy(im26, 'C:/DTEProjektpraktikumSebastianHeid/Vergleichsbilder/img26.raw')