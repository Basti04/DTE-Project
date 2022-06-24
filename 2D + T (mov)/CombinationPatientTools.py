from distutils.command.install_scripts import install_scripts
from multiprocessing.sharedctypes import Value
import numpy as np

from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *


def combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatch):
   # add patient patch and tool patch
    #print(patientPatch.shape)
    #print(toolPatch.shape)
    inputPatch0 = patientPatch + toolPatch
    #writeNumpy(inputPatch0, 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/CTWithout2.raw')
   
    # convert p-value to photon number by using p = -ln(N/N0)
    # sample random N0
    n0 = np.random.uniform(10**3,10**4)
    inputPatch_N = n0 * np.exp(-inputPatch0)

    # sample noise
    #inputPatch_N = np.random.poisson(inputPatch_N)
    for i in range(0,8):
        for j in range(0,384):
            for k in range(0,384):
                if inputPatch_N[i][j][k] == 0:
                    print("pixel = 0")
                    value = n0
                else:
                    value = np.random.poisson(inputPatch_N[i][j][k])
                    # make sure that you do not have a zero
                    while(value <= 0):
                        value = np.random.poisson(inputPatch_N[i][j][k])
                inputPatch_N[i][j][k] = value




    # convert back to p-values
    inputPatch = -np.log(inputPatch_N/n0)

    return inputPatch

# get your projection of patient and tool
def getProjection(filenameToPatients, pathPatients, patientPath, pathStent, pathGuidwire, patchSize):
    
    toolPatch = getToolPatch(pathStent, pathGuidwire, patchSize)
    patientPatch = getPatientPatch(pathPatients, filenameToPatients, patientPath, patchSize)

    
    projection = combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatch)
    


    return projection, toolPatch



# if __name__ == '__main__':
#     pathTools =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
#     pathPatients = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
#     filenameToTools = [name for name in os.listdir(pathTools)]
#     filenameToPatients = [name for name in os.listdir(pathPatients)]
    
   
   




