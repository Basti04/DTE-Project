from torch.utils.data import Dataset
from IOFunctions import *
from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *
from myModel import *
from Dataloader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# pathToolsTraining =  'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
# pathPatientsTraining = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/'
# filenameToTools = [name for name in os.listdir(pathToolsTraining)]
# filenameToPatients = [name for name in os.listdir(pathPatientsTraining)]

# myDataset = DTE(pathToolsTraining, pathPatientsTraining, filenameToTools, filenameToPatients,1, transforms.Compose([]))

# item, _ = myDataset.__getitem__(0) 

# writeNumpy(item.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/CT2.raw')
# writeNumpy(_.detach().numpy(), 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/target2.raw')


# loss_training = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/training_loss2.csv', delimiter=',')
# loss_test = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/savedModels/test_loss2.csv', delimiter=',')
# print(loss_training)
# print(loss_test)