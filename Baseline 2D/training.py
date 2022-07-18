from multiprocessing.dummy import freeze_support
import os
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import time

from myModel import UNET
from Dataloader import DTE

import cProfile, pstats

def train():

    # set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # normalization of the data 
    mean = np.load("C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/mean.npy")
    std = np.load("C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/std.npy")
    myTransforms = transforms.Compose([transforms.Normalize(mean, std)])

    # model 
    myModel = UNET()
    #myModel = torch.load('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/new network/modelLowNoise29_10000Images.pth')
    
    myModel = myModel.to(device)


    # load paths 
    pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
    pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
    pathToolsValidation = 'C:/DTEProjektpraktikumSebastianHeid/validationData/tools/'
    pathPatientsValidation = 'C:/DTEProjektpraktikumSebastianHeid/validationData/patients/'
    filenameToToolsTraining = [name for name in os.listdir(pathToolsTraining)]
    filenameToPatientsTraining = [name for name in os.listdir(pathPatientsTraining)]
    filenameToToolsTest = [name for name in os.listdir(pathToolsValidation)]
    filenameToPatientsTest = [name for name in os.listdir(pathPatientsValidation)]

    # size of datasets
    trainingDatasetSize = 10000
    validationDatasetSize = 2000

    # size of projections
    patchSize = 384
    projectionSize = 1024

    # datasets
    trainingDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToToolsTraining, filenameToPatientsTraining,
                         trainingDatasetSize, True, patchSize, projectionSize)
    testDataset =  DTE(pathToolsValidation, pathPatientsValidation, filenameToToolsTest, filenameToPatientsTest, validationDatasetSize, 
                    True, patchSize,projectionSize)


    # Hyper-parameters 
    cfg = dict()
    cfg['numEpoch'] = 50
    cfg['learning_rate'] = .0001
    cfg['batchSize'] = 5
    # optimizer
    optimizer = torch.optim.Adam(myModel.parameters(), lr=cfg['learning_rate'])
    # loss
    myLoss = nn.L1Loss()

    # save training and test loss
    lossTrain = []
    lossTest = []


    # load data
    myLoader_train = DataLoader(trainingDataset, batch_size=cfg['batchSize'], num_workers=4, pin_memory=True)
    myLoader_test = DataLoader(testDataset, batch_size=cfg['batchSize'], num_workers=4, pin_memory=True)
    N_train = len(trainingDataset) # number of images in the training set

    #N_test = len(testDataset)
    nbr_miniBatch_train = len(myLoader_train) 
    # number of mini-batches
    nbr_miniBatch_test = len(myLoader_test)



    t0 = time.time()

    ep = -1
    for epoch in range(cfg['numEpoch']):
        ep += 1
        print('-- epoch ' + str(epoch))

        running_loss_train = 0.0

        pbar = tqdm(total=(N_train - cfg['batchSize']))

        myModel.train()
        
        for X,y in myLoader_train:
            
            #optimizer.zero_grad()
            for param in myModel.parameters():
                param.grad = None
            # calculate the score and the corresponding loss
            X = X.float()
            X = X.to(device)
            y = y.float()
            y = y.to(device)

            
            score = myModel(X)
            loss = myLoss(score,y)
            
        
            # backpropagation
            loss.backward()
            optimizer.step()
    

        
            # calculate loss and accuracy
            running_loss_train += loss
            #print("End Training")

            pbar.update(cfg['batchSize'])

        # test
        running_loss_test = 0.0

        torch.save(myModel, 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/modelLowNoise'+ str(ep) + '_10000Images.pth')
        # make sure that no training occur
        myModel.eval()
        with torch.no_grad():
            for X,y in myLoader_test:
                X = X.to(device)
                X = X.float()
                y = y.to(device)

                # 1) compute the score and loss
                score = myModel(X)
                loss = myLoss(score, y)
                # 2) estimate the overall loss over the all test set
            
                running_loss_test += loss
    
    
            
        # end epoch
        loss_train = running_loss_train/nbr_miniBatch_train
        loss_test = running_loss_test/nbr_miniBatch_test
    
        print('    loss     (train, test): {:.4f},  {:.4f}'.format(loss_train, loss_test))

        loss_train = loss_train.to("cpu")
        loss_test = loss_test.to("cpu")
        lossTrain.append(loss_train.detach().numpy())
        lossTest.append(loss_test.detach().numpy())
        np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/training_lossLowNoise_10000Images_50.csv', lossTrain, delimiter=',')
        np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/test_lossLowNoise_10000Images_50.csv', lossTest, delimiter=',')


        pbar.close()
    
    
    # the training is over
    tFinal = time.time()
    print('time elapsed = {:.4f} seconds'.format(tFinal-t0))

    # save model and loss
    torch.save(myModel, 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/modelLowNoisefinale_10000Images.pth')
    np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/training_lossLowNoise_10000Images_50.csv', lossTrain, delimiter=',')
    np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/test_lossLowNoise_10000Images_50.csv', lossTest, delimiter=',')

if __name__ =='__main__':
    freeze_support()
    profiler = cProfile.Profile()
    profiler.enable()
    train()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()