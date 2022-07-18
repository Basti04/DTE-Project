import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


from functionsPatientImage import choosePatientProjection
from functionsToolImage import pathRandomToolScene
from CombinationPatientTools import getProjection



def normalize(img, mean, std):
    myTransforms = transforms.Compose([transforms.Normalize(mean[-1], std[-1])])
    normalizedImg = myTransforms(img)
    return normalizedImg

class DTE(Dataset):

    def __init__(self, pathTools, pathPatients, filenameToTools, filenameToPatients, 
                sizeData, normalization, patchSize, projectionSize):

        # load paths to tool and patient files
        self.pathToTools = pathTools
        self.pathToPatients = pathPatients
        self.filenameToTools = filenameToTools
        self.filenameToPatients = filenameToPatients

        # mean and standard deviation of the data set
        self.mean = np.load("C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/mean.npy")
        self.std = np.load("C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/std.npy")

        # decide if data should be normalized or not
        self.normalization = normalization
        # save patch size and projection size
        self.patchSize = patchSize
        self.projectionSize = projectionSize

        # load the paths to tool scenes and to different isocenters of the patient projection
        self.pathStent = ["" for x in range(sizeData)]
        self.pathGuidewire = ["" for x in range(sizeData)]
        self.pathPatient = ["" for x in range(sizeData)]
        for i in range(sizeData):
            self.pathStent[i], self.pathGuidewire[i] = pathRandomToolScene(self.pathToTools,self.filenameToTools)
            self.pathPatient[i] = choosePatientProjection(self.pathToPatients,self.filenameToPatients)
        

    
    def __len__(self):
        return len(self.pathPatient)

    def __getitem__(self, idxPatient):
        idxTool = np.random.randint(0, len(self.pathGuidewire))
        projection, targetTool = getProjection(self.pathPatient[idxPatient],
                                         self.pathStent[idxTool], self.pathGuidewire[idxTool], self.patchSize, 
                                         self.projectionSize)
        projection  = torch.from_numpy(projection)
    
        projection = projection[-1]
        projection = projection.view(1,self.patchSize,self.patchSize)
        if self.normalization == True:
            projection = normalize(projection, self.mean, self.std)
        targetTool = torch.from_numpy(targetTool)
        targetTool = targetTool[-1]
        targetTool = targetTool.view(1,self.patchSize,self.patchSize)
       

        
        return projection, targetTool



# # # # load path 
# import os
# from IOFunctions import writeNumpy
# pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
# pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
# filenameToTools = [name for name in os.listdir(pathToolsTraining)]
# filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]


# # # myModel = UNET()
# myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients, 100, False, 384, 1024)

# for i in range(1):
#     x,_= myDataset.__getitem__(i)

#     writeNumpy(x.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Test Images/input'+str(i)+'.raw')
#     writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Test Images/target'+str(i)+'.raw')



