from tarfile import TarError
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms



from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *


def normalize(img):
    mean = np.load("C:/DTEProjektpraktikumSebastianHeid/mean.npy")
    std = np.load("C:/DTEProjektpraktikumSebastianHeid/std.npy")
    myTransforms = transforms.Compose([transforms.Normalize(mean[0], std[0])])
    normalizedImg = myTransforms(img)
    return normalizedImg

class DTE(Dataset):

    def __init__(self, pathTools, pathPatients, filenameToTools, filenameToPatients, sizeData, normalization, patchSize):

        # load paths to tool and patient files
        self.pathTools = pathTools
        self.pathPatients = pathPatients
        self.filenameToTools = filenameToTools
        self.filenameToPatients = filenameToPatients

        # decide if data should be normalized or not
        self.normalization = normalization
        # save patch size
        self.patchSize = patchSize

        # load the paths 
        self.pathStents = ["" for x in range(sizeData)]
        self.pathGuidwire = ["" for x in range(sizeData)]
        self.patient = ["" for x in range(sizeData)]
        for i in range(sizeData):
            self.pathStents[i], self.pathGuidwire[i] = pathRandomToolScene(pathTools,filenameToTools)
            self.patient[i] = choosePatient(self.pathPatients,self.filenameToPatients)
        

    
    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):
        projection, targetTool = getProjection(self.filenameToPatients, self.pathPatients, self.patient[idx],
                                         self.pathStents[idx], self.pathGuidwire[idx], self.patchSize)
        projection  = torch.from_numpy(projection)
        projection = projection[-1]
        projection = projection.view(1,384,384)
        if self.normalization == True:
            projection = normalize(projection)
        targetTool = torch.from_numpy(targetTool)
        targetTool = targetTool[-1]
        targetTool = targetTool.view(1,384,384)

        
        return projection, targetTool



# # # # load path 
# pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
# pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
# filenameToTools = [name for name in os.listdir(pathToolsTraining)]
# filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]


# # myModel = UNET()
# myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients, 2000, False, 384)

# for i in range(0,20):
#     x,_= myDataset.__getitem__(i)
#     print("load image")
# writeNumpy(x.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/x.raw')
# writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/_.raw')



