#from IOFunctions import *
import shutil
import numpy as np
from random import *

# copy patient S196 to Validation Data
source0 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S196'
dest0 = 'C:/DTEProjektpraktikumSebastianHeid/validationData/patients/S196'
#shutil.copytree(source0, dest0)

# copy patient S193 to Test Data
source1 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S190'
dest1 = 'C:/DTEProjektpraktikumSebastianHeid/testData/patients/S190'
#shutil.copytree(source1, dest1)

# copy patient S163, S164, S166, S181, S182, S185 to Training Data
source2 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S185'
source3 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S182'
source4 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S181'
source5 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S166'
source6 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S164'
source7 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFürSebastian/2022-05-31_patientProjsCount/projs/S163'

dest2 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S185'
dest3 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S182'
dest4 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S181'
dest5 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S166'
dest6 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S164'
dest7 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/patients/S163'

#shutil.copytree(source2, dest2)
#shutil.copytree(source3, dest3)
#shutil.copytree(source4, dest4)
#shutil.copytree(source5, dest5)
#shutil.copytree(source6, dest6)
#shutil.copytree(source7, dest7)

# copy tools

# create array with numbers from 0,...,9999 to keep track which tools are already copied
numbTools = np.ones(10000)
for i in range(0,10000):
    numbTools[i] = int(i)

# loop for copy tools in training data file
for i in range(0,7000):
    rand = randint(0,9999-i)
    num = int(numbTools[rand])
    # delete number so that you do not copy one tool twice
    numbTools = np.delete(numbTools,rand)
    sour01 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    sour02 =  'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    dest01 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    dest02 = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    #shutil.copyfile(sour01,dest01)
    #shutil.copyfile(sour02,dest02)

print(len(numbTools))
# copy tools for validation data
for i in range(0,2000):
    rand = randint(0,2999-i)
    num = int(numbTools[rand])
    # delete number so that you do not copy one tool twice
    numbTools = np.delete(numbTools,rand)
    sour01 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    sour02 =  'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    dest01 = 'C:/DTEProjektpraktikumSebastianHeid/validationData/tools/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    dest02 = 'C:/DTEProjektpraktikumSebastianHeid/validationData/tools/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    #shutil.copyfile(sour01,dest01)
    #shutil.copyfile(sour02,dest02)


# copy tools for test data
for i in range(0,1000):
    rand = randint(0,999-i)
    num = int(numbTools[rand])
    # delete number so that you do not copy one tool twice
    numbTools = np.delete(numbTools,rand)
    sour01 = 'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    sour02 =  'C:/DTEProjektpraktikumSebastianHeid/2022-06-07_DTEDatenFÜrSebastian/2022-06-01_Ngw1-2_Nst1_coupleAllTogether_Nex10000_Nt8_Nthreads2_da0/Results/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    dest01 = 'C:/DTEProjektpraktikumSebastianHeid/testData/tools/ex_'+str(num)+'_guidewire_intLength_1024x1024x8x2.npz'
    dest02 = 'C:/DTEProjektpraktikumSebastianHeid/testData/tools/ex_'+str(num)+'_stent_intLength_1024x1024x8x2.npz'
    #shutil.copyfile(sour01,dest01)
    #shutil.copyfile(sour02,dest02)

print(numbTools)