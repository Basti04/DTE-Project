from distutils.command.install_scripts import install_scripts
from multiprocessing.sharedctypes import Value
import numpy as np

from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *


def combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatch, nbrOfTimesteps):
   # add patient patch and tool patch
    #print(patientPatch.shape)
    #print(toolPatch.shape)
    inputPatch0 = patientPatch + toolPatch
    writeNumpy(inputPatch0, 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D + T/TestImages/toolPatient.raw')
   
    #  initialize an array to store the same projection with different noise
    timeSequenze = np.zeros((nbrOfTimesteps, inputPatch0.shape[1], inputPatch0.shape[2]))
    timeSequenze = timeSequenze.astype(np.float32)
    # convert p-value to photon number by using p = -ln(N/N0)
    # sample random N0
    n0 = np.random.uniform(10**2,10**4)
    inputPatch_N = n0 * np.exp(-inputPatch0)
    

    # sample noise
    #inputPatch_N = np.random.poisson(inputPatch_N)
    print("Start noise sampling")
    for i in range(0,nbrOfTimesteps):
        for j in range(0,timeSequenze.shape[1]):
            for k in range(0,timeSequenze.shape[2]):
                if inputPatch_N[0][j][k] == 0:
                    print("value = 0")
                    value = n0
                else:
                    value = np.random.poisson(inputPatch_N[0][j][k])
                    # make sure that you do not have a zero
                    while(value <= 0):
                        value = np.random.poisson(inputPatch_N[0][j][k])
                timeSequenze[i][j][k] = value

    print("End noise sampling")

    # convert back to p-values
    outputPatch = -np.log(timeSequenze/n0)
    

    return outputPatch

# get your projection of patient and tool
def getProjection(filenameToPatients, pathPatients, patientPath, pathStent, pathGuidwire, patchSize, nbrOfTimesteps):
    toolPatch = getToolPatch(pathStent, pathGuidwire, patchSize)
    patientPatch = getPatientPatch(pathPatients, filenameToPatients, patientPath, patchSize)
    projection = combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatch, nbrOfTimesteps)
    return projection, toolPatch



# if __name__ == '__main__':
#     pathTools =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
#     pathPatients = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
#     filenameToTools = [name for name in os.listdir(pathTools)]
#     filenameToPatients = [name for name in os.listdir(pathPatients)]

    
   
   




