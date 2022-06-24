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

    def __init__(self, pathTools, pathPatients, filenameToTools, filenameToPatients, sizeData, normalization, patchSize, nbrOftimesteps):

        # load paths to tool and patient files
        self.pathTools = pathTools
        self.pathPatients = pathPatients
        self.filenameToTools = filenameToTools
        self.filenameToPatients = filenameToPatients

        # decide if data should be normalized or not
        self.normalization = normalization
        # save patch size
        self.patchSize = patchSize
        self.nbrOfTimesteps = nbrOftimesteps

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
                                         self.pathStents[idx], self.pathGuidwire[idx], self.patchSize, self.nbrOfTimesteps)
        projection  = torch.from_numpy(projection)
        print(projection.shape)
        projection = projection.view(8,384,384)
        if self.normalization == True:
            projection = normalize(projection)
        targetTool = torch.from_numpy(targetTool)
        targetTool = targetTool.view(1,384,384)

        
        return projection, targetTool



#load path 
pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
filenameToTools = [name for name in os.listdir(pathToolsTraining)]
filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]


# # # myModel = UNET()
myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients, 2, False, 384, 8)

x,_= myDataset.__getitem__(1)
print(x.shape)

writeNumpy(x.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D + T/TestImages/x.raw')
writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D + T/TestImages/_.raw')



