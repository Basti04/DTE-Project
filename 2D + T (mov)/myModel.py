import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


from IOFunctions import *
from functionsPatientImage import *
from functionsToolImage import *
from CombinationPatientTools import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv(in_channels, out_channels, kernel=3):
    downsample = []
    downsample.append(nn.Conv2d(in_channels, out_channels, kernel))
    downsample.append(nn.Dropout2d(0.5))
    downsample.append(nn.LeakyReLU())
    downsample = nn.Sequential(*downsample)
    return downsample

def upwards(in_channels, out_channels, kernel=3):
    upsample = []
    upsample.append(nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2))
    upsample.append(nn.LeakyReLU())
    upsample = nn.Sequential(*upsample)
    return upsample

# define maxpooling
def MaxPooling():
    maxpool = []
    maxpool.append(nn.MaxPool2d(2))
    maxpool = nn.Sequential(*maxpool)
    return maxpool



class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__() 
         # encoder path
        self.d1 = conv(8,64)
        self.d2 = conv(64,64)
        self.d3 = MaxPooling()

        self.d4 = conv(64,128)
        self.d5 = conv(128,128)
        self.d6 = MaxPooling()

        self.d7 = conv(128,256)
        self.d8 = conv(256,256)
        self.d9 = MaxPooling()

        self.d10 = conv(256,512)
        self.d11 = conv(512,512)
        self.d12 = MaxPooling()

        self.d13 = conv(512,1024)
        self.d14 = conv(1024,1024)
        

        # decoder path
        self.u1 = upwards(1024,512)
    
        self.u2 = conv(1024,512)
        self.u3 = conv(512,512)
        self.u4 = upwards(512,256)

        self.u5 = conv(512,256)
        self.u6 = conv(256,256)
        self.u7 = upwards(256,128)

        self.u8 = conv(256,128)
        self.u9 = conv(128,128)
        self.u10 = upwards(128,64)

        self.u11 = conv(128,64)
        self.u12 = conv(64,64)
        self.u13 = conv(64,1,(1,1))

        
    
    def forward(self, img):
        # encoder path
        d1 = self.d1(img)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        #print("d1",d1.shape)
       # print("d2",d2.shape)
        #print("d3",d3.shape)

        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        #print("d4",d4.shape)
        #print("d5",d5.shape)
        #print("d6",d6.shape)

        d7 = self.d7(d6)
        d8 = self.d8(d7)
        d9 = self.d9(d8)
        #print("d7",d7.shape)
        #print("d8",d8.shape)
        #print("d9",d9.shape)

        d10 = self.d10(d9)
        d11 = self.d11(d10)
        d12 = self.d12(d11)
        #print("d10",d10.shape)
        #print("d11",d11.shape)
        #print("d12",d12.shape)

        d13 = self.d13(d12)
        d14 = self.d14(d13)
        #print("d13",d13.shape)
        #print("d14",d14.shape)


        # decoder path
        u1 = self.u1(d14)
        #print("u1",u1.shape)

        # concatenation of encoder and decoder path
        if u1.shape != d11.shape:
            u1 = TF.resize(u1, size=d11.shape[2:])
        u1 = torch.cat((d11, u1), dim=1)
        #print("u1",u1.shape)

        u2 = self.u2(u1)
        u3 = self.u3(u2)
        u4 = self.u4(u3)
        #print("u2",u2.shape)
        #print("u3",u3.shape)
        #print("u4",u4.shape)

        # concatenation of encoder and decoder path
        if u4.shape != d8.shape:
            u4 = TF.resize(u4, size=d8.shape[2:])
        u4 = torch.cat((d8, u4), dim=1)
        #print("u4",u4.shape)

        u5 = self.u5(u4)
        u6 = self.u6(u5)
        u7 = self.u7(u6)
        #print("u5",u5.shape)
        #print("u6",u6.shape)
        #print("u7",u7.shape)

        # concatenation of encoder and decoder path
        if u7.shape != d5.shape:
            u7 = TF.resize(u7, size=d5.shape[2:])
        u7 = torch.cat((d5, u7), dim=1)
        #print("u7",u7.shape)

        u8 = self.u8(u7)
        u9 = self.u9(u8)
        u10 = self.u10(u9)
        #print("u8",u8.shape)
        #print("u9",u9.shape)
        #print("u10",u10.shape)

        # concatenation of encoder and decoder path
        if u10.shape != d2.shape:
            u10 = TF.resize(u10, size=d2.shape[2:])
        u10 = torch.cat((d2, u10), dim=1)
        #print("u10",u10.shape)

        u11 = self.u11(u10)
        u12 = self.u12(u11)
        #print("u11",u11.shape)
        #print("u12",u12.shape)
        u13 = self.u13(u12)
        #print("u13",u13.shape)
        if u13.shape != img.shape:
            u13 = TF.resize(u13, size=img.shape[2:])
        #print("u13", u13.shape)
        
        return u13


#model = UNET()


#count = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Number of parameters:",count)

# CTImage = getImage()
# CTImage = torch.from_numpy(CTImage)
# CTImage = CTImage.view(1,8,384,384)

# model = UNET()
# model = model.float()

# target = model(CTImage)
# print(target.shape)

