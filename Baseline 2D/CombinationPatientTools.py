from multiprocessing.sharedctypes import Value
import numpy as np
import scipy

from functionsPatientImage import getPatientPatch
from functionsToolImage import getToolPatch
from IOFunctions import writeNumpy


def combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatch):
    #print("combinePatientToolProjectionAndNoiseSampling")
   # add patient patch and tool patch
    inputPatch = patientPatch + toolPatch
    #writeNumpy(inputPatch, 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/test Data 30/inputWithoutNoise_384x384x1.raw')
   
    # convert p-value to photon number by using p = -ln(N/N0)
    # sample random n0
    #n0 = np.random.uniform(10**3,5*10**3)
    ##
    n0 = np.random.uniform(1.25*10**2,2.5*10**2)
    ##
    #n0 = 10**4
    inputPatch_N = n0 * np.exp(-inputPatch)

    # sample noise
    std = np.random.uniform(0.3,1)
    inputPatch_N = inputPatch_N + scipy.ndimage.gaussian_filter((np.random.poisson(inputPatch_N) - inputPatch_N)[-1], std)
    
    
    #inputPatch_N = np.maximum(inputPatch_N,1,)
    inputPatch_N[inputPatch_N <= 0] = 1

    # convert back to p-values
    inputPatch = -np.log(inputPatch_N/n0)
    inputPatch = inputPatch.astype("float32")

    return inputPatch

# get your projection of patient and tool
def getProjection(patientPath, pathStent, pathGuidwire, patchSize, projectionSize):
    
    toolPatchFilterd, toolPatch = getToolPatch(pathStent, pathGuidwire, patchSize)
    patientPatch = getPatientPatch(patientPath, patchSize, projectionSize)
    projection = combinePatientToolProjectionAndNoiseSampling(patientPatch, toolPatchFilterd)

    return projection, toolPatch



# if __name__ == '__main__':
#     pathTools =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
#     pathPatients = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
#     filenameToTools = [name for name in os.listdir(pathTools)]
#     filenameToPatients = [name for name in os.listdir(pathPatients)]
    
   
   




