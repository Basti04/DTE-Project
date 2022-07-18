from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

# ## plot evolution of the loss
# lossTrain1 = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/4-Training 40 Epochen 10000 Trainingsbilder/train_lossLowNoise_10000Images.csv')
# lossTest1 = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/4-Training 40 Epochen 10000 Trainingsbilder/test_lossLowNoise_10000Images.csv')

# lossTrain2 = np.array([0.0347, 0.0347, 0.0345, 0.0345, 0.0345, 0.0344, 0.0344, 0.0343, 0.0343, 0.0341, 0.0338, 0.0338, 0.0342, 0.0339, 0.0340, 0.0337 ])
# lossTest2 = np.array([0.0308, 0.0307, 0.0312, 0.0301, 0.0317, 0.0308, 0.0306, 0.0296, 0.0297, 0.0297, 0.0309, 0.0309, 0.0305, 0.0306, 0.0306, 0.0305])

# lossTrain3 = np.array([0.0363, 0.0360, 0.0356, 0.0352, 0.0349, 0.0348, 0.0343, 0.0342,0.0339, 0.0340, 0.0336, 0.0336, 0.0333, 0.0332, 0.0330, 0.0328])
# lossTest3 = np.array([0.0307, 0.0317, 0.0310, 0.0303, 0.0303, 0.0308, 0.0299, 0.0285, 0.0286, 0.0299, 0.0302, 0.0287, 0.0286, 0.0307, 0.0288, 0.0312])

# lossTrain = np.concatenate([lossTrain1, lossTrain2, lossTrain3])
# lossTest = np.concatenate([lossTest1, lossTest2, lossTest3])

# np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/4-Training 40 Epochen 10000 Trainingsbilder/trainingLoss_50epochs_10000Images.csv', lossTrain)
# np.savetxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/4-Training 40 Epochen 10000 Trainingsbilder/testLoss_50epochs_10000Images.csv', lossTest)

# load loss
#lossTrain = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/9-Training 50 Epochen 10000Training 2000Test mehr noise/training_lossLowNoise_10000Images.csv')
#lossTest = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D/savedModels/9-Training 50 Epochen 10000Training 2000Test mehr noise/test_lossLowNoise_10000Images.csv')

#lossTrain0 = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/new network/training_lossLowNoise_10000Images.csv')
#lossValidation0 =  np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Try/new network/test_lossLowNoise_10000Images.csv')
lossTrain = np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/training_lossLowNoise_10000Images_50.csv')
lossValidation =  np.loadtxt('C:/DTEProjektpraktikumSebastianHeid/Baseline 2D Attempt 4/network 1.25_10^2 to 2.5_10^2/test_lossLowNoise_10000Images_50.csv')

#lossTrain = np.concatenate([lossTrain0, lossTrain1[:-2]])
#lossValidation = np.concatenate([lossValidation0, lossValidation1[:-2]])

# number of epochs
epochs = np.linspace(0,len(lossTrain),len(lossTrain))
print(len(epochs))

plt.plot(epochs, lossTrain, label="Training loss")
plt.plot(epochs, lossValidation, label="Validation loss")
plt.title("Evolution of the loss for 50 epochs")

plt.legend()
plt.show()