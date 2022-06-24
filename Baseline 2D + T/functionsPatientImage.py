from pickletools import float8
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from random import *


from IOFunctions import *

# choos random patient, random isocenter
def choosePatient(pathPatients,filenamesToPatients):
    # choose random patient
    nbrOfPatients = len(filenamesToPatients)
    randomPatient = randint(0,nbrOfPatients-1)

    # choos random isocenter 
    pathPatientImage = pathPatients + filenamesToPatients[randomPatient]

    # only consider the raw files
    ending = '.raw' 
    imgsPatient = [img for img in os.listdir(pathPatientImage) if img.endswith(ending)]
    nbrOfImg = len(imgsPatient)
    randomImage = randint(0,nbrOfImg-1)

    patientPath = pathPatientImage + '/' + imgsPatient[randomImage]

    return patientPath


def loadPatientImage(patientPath):
    #laod the image w
    imagePatient = readNumpy(patientPath)

    # choose random angle -> there are 72 different angles
    randomAngle = np.random.randint(imagePatient.shape[0])

    # choose the angle 
    imagePatient = imagePatient[randomAngle]

    return imagePatient
    

# choose random patch from the projection of the patient (384^2)
def choosePatientPatch(imagePatient, patchSize):
    # sample the coordinates of the upper left corner of the patch 
    x = randint(0,639)
    y = randint(0,639)
    img = imagePatient[y:y+patchSize, x:x+patchSize]

    return img 

def getPatientPatch(pathPatients, filenameToPatients, patientPath, patchSize):
    #patientPath = choosePatient(pathPatients, filenameToPatients)
    patientImage = loadPatientImage(patientPath)
    patientPatch = choosePatientPatch(patientImage, patchSize)
    
    return patientPatch


  

# if __name__ == '__main__':
#     patientPatch = getPatientPatch()
    

   
    
    




