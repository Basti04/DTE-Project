import os
from random import randint
from IOFunctions import readNumpyPatient

# choos random patient, random isocenter
def choosePatientProjection(pathPatients,filenamesToPatients):
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


def loadRandomPatientProjection(patientPath, nbrOfPixels):
    Nproj = patientPath.split('/')[-1].split('_')[-1].split('x')[-1].split('.')[0]
    nproj = int(Nproj)
    # random projection of the patient 
    nproj = randint(0, nproj-1)

    count = nbrOfPixels**2
    offset = 4 * nbrOfPixels**2 * nproj
    imagePatient = readNumpyPatient(patientPath, count, offset)

    return imagePatient
    

# choose random patch from the projection of the patient (384^2)
def choosePatientPatch(imagePatient, patchSize, projectionSize):
    #print("ChoosePatientPatch")
    # sample the coordinates of the upper left corner of the patch 
    x = randint(0,projectionSize-patchSize-1)
    y = randint(0,projectionSize-patchSize-1)
    img = imagePatient[:,y:y+patchSize, x:x+patchSize]

    return img 

def getPatientPatch(patientPath, patchSize,projectionSize):
    patientProjection = loadRandomPatientProjection(patientPath, projectionSize)
    patientPatch = choosePatientPatch(patientProjection, patchSize, projectionSize)
    return patientPatch


  

# if __name__ == '__main__':
#     patientPatch = getPatientPatch()
    

   
    
    




