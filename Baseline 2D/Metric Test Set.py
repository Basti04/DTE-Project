import numpy as np
from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *
from Dataloader import *
import torch.nn as nn


pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/testData/tools/'
pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/testData/patients/'
filenameToTools = [name for name in os.listdir(pathToolsTraining)]
filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]#


myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients,3000, True, 384,1024)

model = torch.load('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 2/network 8_10^3 to 16_10^3/modelLowNoise49_10000Images.pth')
model.to("cpu")

metric = nn.L1Loss()
running_loss = 0
for i in range(3000):
    print(i)
    item, _ = myDataset.__getitem__(i) 
    item.to("cpu")
    item = item.float() 
    item = item.view(1,1,384,384)
    _ = _.view(1,1,384,384)
    pred = model(item)
    running_loss += metric(pred, _)

loss = running_loss/3000


torch.save(loss, 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 2/network 8_10^3 to 16_10^3/test Set loss.pth')
    
