from multiprocessing.dummy import freeze_support
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *
from Dataloader import *

def train():

    # set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # normalization of the data 
    mean = np.load("C:/DTEProjektpraktikumSebastianHeid/mean.npy")
    std = np.load("C:/DTEProjektpraktikumSebastianHeid/std.npy")
    myTransforms = transforms.Compose([transforms.Normalize(mean, std)])

    # model 
    myModel = UNET()
    myModel = myModel.to(device)


    # load path 
    pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
    pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
    pathToolsTest = 'C:/DTEProjektpraktikumSebastianHeid/testData/tools/'
    pathPatientsTest = 'C:/DTEProjektpraktikumSebastianHeid/testData/patients/'
    filenameToToolsTraining = [name for name in os.listdir(pathToolsTraining)]
    filenameToPatientsTraining = [name for name in os.listdir(pathPatientsTraining)]
    filenameToToolsTest = [name for name in os.listdir(pathToolsTest)]
    filenameToPatientsTest = [name for name in os.listdir(pathPatientsTest)]

    # size of datasets
    trainingDatasetSize = 50
    testDatasetSize = 10

    # datasets
    trainingDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToToolsTraining, filenameToPatientsTraining,
                         trainingDatasetSize, True, 384)
    testDataset =  DTE(pathToolsTest, pathPatientsTest, filenameToToolsTest, filenameToPatientsTest, testDatasetSize, 
                    False, 384)


    # Hyper-parameters 
    cfg = dict()
    cfg['numEpoch'] = 10
    cfg['learning_rate'] = .0001
    cfg['batchSize'] = 2
    # optimizer
    optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])
    # loss
    myLoss = nn.MSELoss()

    # save training and test loss
    lossTrain = []
    lossTest = []


    # load data
    myLoader_train = DataLoader(trainingDataset, batch_size=cfg['batchSize'], num_workers=2, pin_memory=True)
    myLoader_test = DataLoader(testDataset, batch_size=cfg['batchSize'], num_workers=2, pin_memory=True)
    N_train = len(trainingDataset) # number of images in the training set

    N_test = len(testDataset)
    nbr_miniBatch_train = len(myLoader_train) # number of mini-batches
    nbr_miniBatch_test = len(myLoader_test)



    t0 = time.time()
    for epoch in range(cfg['numEpoch']):
        print('-- epoch ' + str(epoch))

        running_loss_train = 0.0

        pbar = tqdm(total=(N_train - cfg['batchSize']))

        myModel.train()
        for X,y in myLoader_train:
        
            optimizer.zero_grad()
            # calculate the score and the corresponding loss
        
            X = X.to(device)
            y = y.to(device)
            score = myModel(X)
            loss = myLoss(score,y)
        
            # backpropagation
            loss.backward()
            optimizer.step()

        
            # calculate loss and accuracy
            running_loss_train += loss

            pbar.update(cfg['batchSize'])

        # test
        running_loss_test = 0.0
        # make sure that no training occur
        myModel.eval()
        with torch.no_grad():
            for X,y in myLoader_test:
                X = X.to(device)
                y = y.to(device)

                # 1) compute the score and loss
                score = myModel(X)
                loss = myLoss(score, y)
                # 2) estimate the overall loss over the all test set
            
                running_loss_test += loss
    
    
            
        # end epoch
        # c.3) statistics
        loss_train = running_loss_train/nbr_miniBatch_train
        loss_test = running_loss_test/nbr_miniBatch_test
    
        print('    loss     (train, test): {:.4f},  {:.4f}'.format(loss_train, loss_test))

        loss_train = loss_train.to("cpu")
        loss_test = loss_test.to("cpu")
        lossTrain.append(loss_train.detach().numpy())
        lossTest.append(loss_test.detach().numpy())

        pbar.close()
    
    
    # the training is over
    tFinal = time.time()
    print('time elapsed = {:.4f} seconds'.format(tFinal-t0))

    # save model and loss
    torch.save(myModel, 'C:/DTEProjektpraktikumSebastianHeid/savedModels/modelLowNoise.pth')
    np.savetxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/training_lossLowNoise.csv', lossTrain, delimiter=',')
    np.savetxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/test_lossLowNoise.csv', lossTest, delimiter=',')

if __name__ =='__main__':
    freeze_support()
    train()