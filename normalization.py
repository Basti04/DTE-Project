import numpy as np


from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *
from Dataloader import *

# calculate the mean and the standard deviation of 500 images
# use the results for normalization of the image in the dataloader

myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients, 500)

mean = torch.zeros((8,384,384))
std = torch.zeros((8, 384, 384))

i = 0
j = 0
# calculate the mean of each pixel
for  CTImage, target in myDataset:
    print("mean: ",i)
    mean += CTImage
    i += 1

# claculate the standard deviation of each pixel
for CTImage, target in myDataset:
    print("std: ", j)
    std += (CTImage - mean)**2/myDataset.__len__()
    j += 1

std = torch.sqrt(std)

mean = mean/myDataset.__len__()

np.save("C:/DTEProjektpraktikumSebastianHeid/mean", mean )
np.save("C:/DTEProjektpraktikumSebastianHeid/std", std )

