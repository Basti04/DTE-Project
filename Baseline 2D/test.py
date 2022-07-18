from IOFunctions import *
from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *
from Dataloader import *
from training import * 


import cProfile, pstats

def t():
    pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
    pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
    filenameToTools = [name for name in os.listdir(pathToolsTraining)]
    filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]

    myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients,100, True, 384,1024)
    #modelNew = UNET()
    modelNew = torch.load('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/new network2/modelLowNoise49_10000Images.pth')
    modelNew.to("cpu")
    #modelNew2 = torch.load('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/new network/modelLowNoise30_10000Images.pth')
    #modelNew2.to("cpu")
#     for i in range(100):
    
    



    for i in range(10):
        print(i)
        item, _ = myDataset.__getitem__(i) 
        item.to("cpu")
        item = item.float() 
        item = item.view(1,1,384,384)
        pred = modelNew(item)
        #pred2 = modelNew2(item)
        
        writeNumpy(item.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/test Data 30/input'+str(i) + '_384x384x1.raw')
        writeNumpy(pred.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/test Data 30/predictionL1'+str(i) + '_384x384x1.raw')
        writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/test Data 30/target'+str(i) + '_384x384x1.raw')
        #writeNumpy(pred2.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/test Data 30/predictionL2'+str(i) + '_384x384x1.raw')


t()

#prediction = readNumpy('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/test Data 30/prediction4_384x384x1.raw')
# target = readNumpy('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/test Data 30/target4_384x384x1.raw')

# myLoss = nn.MSELoss(reduction='sum')

# prediction = torch.from_numpy(prediction)
# target = torch.from_numpy(target)

# l = myLoss(prediction, target)

# print(l)










# if __name__ =='__main__':
#     profiler = cProfile.Profile()
#     profiler.enable()
#     t()
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('tottime')
#     stats.print_stats()

# loss_training = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/training_loss2.csv', delimiter=',')
# loss_test = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/test_loss2.csv', delimiter=',')
# print(loss_training)
# print(loss_test)



# import matplotlib.pyplot as plt

# trainLoss = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/2-Training 20 Epochen 10000 Trainingsbilder/training_lossLowNoise1_10000Images.csv')
# testLoss = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/2-Training 20 Epochen 10000 Trainingsbilder/test_lossLowNoise1_10000Images.csv')

# xfine = np.linspace(0,19,20)
# print(xfine)

# plt.plot(xfine, trainLoss, label="Train loss")
# plt.title("Training loss")
# plt.plot(xfine, testLoss, label="Test loss")
# plt.title("Loss")
# plt.legend()
# plt.show()



# model = torch.load('C:/DTEProjektpraktikumSebastianHeid/savedModels/modelLowNoise1_10000Images.pth')
# # # # load path 
# pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
# pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
# filenameToTools = [name for name in os.listdir(pathToolsTraining)]
# filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]


# # # myModel = UNET()
# projectionSize = 1024
# myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients, 2000, False, 384, projectionSize)

# x,_= myDataset.__getitem__(0)

# pred = model(x)
# writeNumpy(x.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/prediction2.raw')
# writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/target2.raw')


# ----------------------------------
# ------ gaussian filtering --------
# ----------------------------------


# # gaussian 
# def gaussian(x, mu, std):
#     #print("gaussian")
#     gaus = 1/np.sqrt(2 * np.pi * std**2) * np.exp(-(x-mu)**2/(2 * std**2))
#     return gaus

# # gaussian kernel
# def gaussian_kernel(size, std, mu=0):
#     #print("gaussian_kernel")
#     row = np.linspace(-(size//2), size//2, size)
#     #print(row)
#     row_gaus = gaussian(row, mu, std)
#     matrix_gaus = np.outer(row_gaus, row_gaus)
#     #print(matrix_gaus)
    
#     return matrix_gaus    

# def convolutionGaussian(tool, size_kernel):
#     #print("ConvolutionGaussian")
#     # get kernel
#     tool = torch.from_numpy(tool)
#     conv = torch.zeros((8,1006,1006))
    
#     for i in range(conv.shape[0]):
#         #std = randint(100,200)
#         std = 2
#         kernel = gaussian_kernel(size_kernel,std)
#         kernel = torch.from_numpy(kernel)
#         kernel = kernel.float()
#         kernel = kernel.view(1,1,size_kernel,size_kernel).repeat(1, 1, 1, 1)
#         tool_ = tool[i]
#         tool_ = tool_.view(1,1,tool_.shape[0],tool_.shape[1])
#         x= F.conv2d(tool_, kernel)
#         conv[i] = x[0][0]
    
#     conv = conv.detach().numpy()
#     #print("ConvDone")
#     return conv