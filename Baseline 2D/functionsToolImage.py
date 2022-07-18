import numpy as np
from random import randint
from scipy import ndimage
from IOFunctions import writeNumpy
from IOFunctions import loadNpz,loadNpzTool
import scipy



# load tool scene
def pathRandomToolScene(pathTools, filenameToTools):
    #choose random scene 
    randNbr = randint(0, len(filenameToTools) - 1)
    splitFilename = filenameToTools[randNbr].split('_')

    # if the scene is a stent, then load the corresponding guidewire scene 
    if (splitFilename[2]== 'stent'):
        pathStent = pathTools + '/' + filenameToTools[randNbr]
        splitFilename[2] = 'guidewire'
        filenameToGuidewire = "_".join(splitFilename)
        pathGuidewire = pathTools + '/' + filenameToGuidewire

    # if the scene is a guidewire, then load the corresponding stent scene
    else:
        pathGuidewire = pathTools + '/' + filenameToTools[randNbr]
        splitFilename[2] = 'stent'
        filenameToStent = "_".join(splitFilename)
        pathStent =  pathTools + '/' + filenameToStent

    return pathStent, pathGuidewire


def loadToolScene(pathStent, pathGuidewire):
    #load guidwire and stent
    stent = loadNpzTool(pathStent) 
    guidewire = loadNpzTool(pathGuidewire)

    # only using the first projection
    stentSequence = stent[0]
    guidewireSequence = guidewire[0]
    #print(guidewireSequence.shape)

    return guidewireSequence, stentSequence
  

# convert intersection lengths to projection values
def intersectionLengthToProjectionValue(guidewire, stent):
    # the intersection length is given in mm
    # 0.4
    attenuationCoefficient = 0.65
    #attenuationCoefficient = 10
    stent = stent * attenuationCoefficient
    guidewire = guidewire * attenuationCoefficient
    return guidewire, stent


# adding stent and tool projection
def combineToolProjections(guidewire, stent):
    tool = guidewire + stent
    return tool


# calculate the center of mass of one patch
def centerOfMass(tool):
    # add the different time steps of the tools
    t = np.sum(tool, axis=0)
    coord = ndimage.measurements.center_of_mass(t)
    y,x = int(coord[0]), int(coord[1])
    return x, y


# choose random patch from tool projection (being not too empty)
def chooseRandomPatchTool(tool, patchSize):
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

    # make sure that x+patchSize_x<= Nx  and y + patchSize_y<= Ny because otherwise you cannot get a patch of the 
    # the size patchSize
     
    if(x>=tool.shape[2] - patchSize):
        x = randint( x_cm - patchSize, tool.shape[2] - patchSize- 1)
    if(y>=tool.shape[1] - patchSize):
        y = randint(y_cm - patchSize, tool.shape[1] - patchSize- 1)

    img = tool[0:8, y:y+patchSize, x:x+patchSize]
    return img 

def gaussianFilter(toolPatch):
    #sample standard deviation
    std = np.random.uniform(0.4,0.6)
    filteredToolProjection = scipy.ndimage.gaussian_filter(toolPatch[-1], std)
    return filteredToolProjection
    #return toolPatch

    

def getToolPatch(pathStent, pathGuidwire, patchSize):
    guidewire, stent = loadToolScene(pathStent, pathGuidwire)
    guidewire, stent = intersectionLengthToProjectionValue(guidewire, stent)
    toolProjection = combineToolProjections(guidewire, stent)
    toolPatch = chooseRandomPatchTool(toolProjection, patchSize)
    filteredToolPatch = gaussianFilter(toolPatch)
    return filteredToolPatch, toolPatch









# pathguidewire = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_0_guidewire_intLength_1024x1024x8x2.npz'
# pathstent = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/ex_0_stent_intLength_1024x1024x8x2.npz'
# img = getToolPatch(pathstent, pathguidewire, 384)



#tool = getToolPatch(pathstent, pathguidewire, 384)
#patch = getToolPatch(pathstent, pathguidewire, 384)
#stent = loadNpz(pathstent)
#writeNumpy(img,'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/filteredImg.raw' )
#patch = convolutionGaussian(stent, 19)
#patch = patch.detach().numpy()
#print(patch.shape)

#writeNumpy(tool, 'C:/DTEProjektpraktikumSebastianHeid/BeispielNoise/conv.raw')
    
   
# pathTools = 'C:/DTEProjektpraktikumSebastianHeid/trainingData/tools/'
# filenameTools = [name for name in os.listdir(pathTools)]
# pathRandomToolScene(pathTools, filenameTools)