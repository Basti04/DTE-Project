from unittest.mock import patch
import torch
import torch.nn.functional as F

import numpy as np
from random import *
from scipy import ndimage
import os
from IOFunctions import *



# load tool scene
def pathRandomToolScene(pathTools, filenameToTools):
    # choose random scene index
    randNbr = randint(0,(len(filenameToTools)-2)/2)
    nbr = randNbr * 2
    path_stent = pathTools + '/' + filenameToTools[nbr]
    path_guidwire = pathTools + '/' + filenameToTools[nbr + 1]
    return path_stent, path_guidwire

def loadRandomToolScene(pathStent, pathGuidwire):
    #load guidwire and stent
    stent = loadNpz(pathStent) 
    guidwire = loadNpz(pathGuidwire)

    # only using the first projection
    stentSequenz = stent[0]
    guidwireSequenz = guidwire[0] 

    return guidwireSequenz, stentSequenz



# convert intersection lengths to projection values
def converter(guidwire, stent):
    # the intersection length is given in mm
    #attenuationCoefficient = 0.4
    attenuationCoefficient = 4
   
    stent = stent * attenuationCoefficient
    guidwire = guidwire * attenuationCoefficient
    #print("Converter Done!!")
    return guidwire, stent

# gaussian 
def gaussian(x, mu, std):
    gaus = 1/np.sqrt(2 * np.pi * std**2) * np.exp(-(x-mu)**2/(2 * std**2))
    return gaus

# gaussian kernel
def gaussian_kernel(size, std, mu=0):
    row = np.linspace(-(size//2), size//2, size)
    #print(row)
    row_gaus = gaussian(row, mu, std)
    matrix_gaus = np.outer(row_gaus, row_gaus)
    #print(matrix_gaus)
    
    return matrix_gaus    

def convolutionGaussian(tool, size_kernel):
    # get kernel
    tool = torch.from_numpy(tool)
    conv = torch.zeros((8,1006,1006))
    
    for i in range(conv.shape[0]):
        #std = randint(100,200)
        std = 2
        kernel = gaussian_kernel(size_kernel,std)
        kernel = torch.from_numpy(kernel)
        kernel = kernel.float()
        kernel = kernel.view(1,1,size_kernel,size_kernel).repeat(1, 1, 1, 1)
        tool_ = tool[i]
        tool_ = tool_.view(1,1,tool_.shape[0],tool_.shape[1])
        x= F.conv2d(tool_, kernel)
        conv[i] = x[0][0]
    
    conv = conv.detach().numpy()
    #print("ConvDone")
    return conv


# adding stent and tool projection
def getToolProjection(guidwire, stent):
    tool = guidwire + stent
    return tool


# calculate the center of mass of one patch
def centerOfMass(tool):
    # add the different time steps of the tools
    t = np.sum(tool, axis=0)
    coord = ndimage.measurements.center_of_mass(t)
    x,y = int(coord[0]), int(coord[1])
    return x, y


# choose random patch from tool projection (being not too empty)
def chooseRandomPatchTool(tool, patchSize):
    #print("Choose Tool Patch!!")
    # determin centor of mass
    x_cm, y_cm = centerOfMass(tool)

    # sample coordinates of the upper left corner of the patch (first time)
    # make sure that  you only get positive values for x and y
    if x_cm < patchSize:
        x = randint(0, x_cm)
    else:
        x = randint( x_cm - patchSize,   x_cm)
    
    if y_cm < patchSize:
        y = randint(0, y_cm)
    else:
        y = randint( y_cm - patchSize,  y_cm)

    # make sure that x<= tool.shape[1] - patchSize and y<= tool.shape[2] - patchSize because otherwise you cannot get a patch of the 
    # having the size patchSize
     
    if(x>tool.shape[1] - patchSize):
        x = randint( x_cm - patchSize, tool.shape[1] - patchSize)
    if(y>tool.shape[2] - patchSize):
        y = randint(y_cm - patchSize, tool.shape[2] - patchSize)

    img = tool[0:8, y:y+patchSize, x:x+patchSize]
    return img 

    

def getToolPatch(pathStent, pathGuidwire, patchSize):
    guidwire, stent = loadRandomToolScene(pathStent, pathGuidwire)
    guidwire, stent = converter(guidwire, stent)
    tool = getToolProjection(guidwire, stent)
    tool = convolutionGaussian(tool,19)
    patch = chooseRandomPatchTool(tool, patchSize)
    print(patch.shape)
    return patch

#pathguidwire = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_0_guidewire_intLength_1024x1024x8x2.npz'
#pathstent = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_0_stent_intLength_1024x1024x8x2.npz'


#tool = getToolPatch(pathstent, pathguidwire, 384)
#patch = getToolPatch(pathstent, pathguidwire, 384)
#stent = loadNpz(pathstent)
#writeNumpy(stent,'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/org.raw' )
#patch = convolutionGaussian(stent, 19)
#patch = patch.detach().numpy()
#print(patch.shape)

#writeNumpy(tool, 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/conv.raw')
    
   
