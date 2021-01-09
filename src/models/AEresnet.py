import os
import time
import torch
import random
import numpy as np

from torch import nn
from torch.nn import functional as F


#convolutional block defination
class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
                nn.Conv2d(**kwargs),
                nn.BatchNorm2d(num_features=kwargs['out_channels']),       
        )
        
    def forward(self,x):
        return self.block(x)
        

#residual block defination
class ResBlock(nn.Module):
    def __init__(self,in_channels, out_channels, stride = 1,downsample = False):
        super(ResBlock,self).__init__()
        
        self.block1 = ConvBlock(in_channels= in_channels,out_channels= out_channels, stride= stride,
                                kernel_size=3, padding = 1, bias= False)
        self.block2 = ConvBlock(in_channels= out_channels,out_channels= out_channels, stride= 1,
                               kernel_size= 3,padding = 1, bias = False)
        
        self.relu = nn.ReLU(inplace=True)
        
        if downsample:
            self.downsample = ConvBlock(in_channels=in_channels,out_channels= out_channels,
                                        kernel_size= 1, stride= 2, bias= False )
        else:
            self.downsample = None
        
        
        
        
    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.relu(x)
        x = self.block2(x)
        
        if self.downsample != None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        
        return x


#ResNet Architecture Defination
class AEResnet(nn.Module):
    def __init__(self, output_dim=2, res34=False,):
        super(AEResnet,self).__init__()
        
        self.output_dim = output_dim
        self.res34 = res34
        self.same_layer = nn.MaxPool2d(kernel_size=1,stride=1)
        
        
        
        self.conv1 = ConvBlock(in_channels= 3, out_channels= 64, kernel_size= 7,
                        stride= 2, padding= 3, bias= False) 

        self.max_pool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding=1)
        self.relu = nn.ReLU(inplace= True)
        
        
        self.conv2_x = nn.Sequential(
                ResBlock(in_channels=64, out_channels=64, stride=1, downsample=False),
                ResBlock(in_channels=64, out_channels=64, stride=1, downsample=False),
                ResBlock(in_channels=64, out_channels=64, stride=1, downsample=False) 
                        if self.res34 else self.same_layer
        
        ) 
        self.conv3_x = nn.Sequential(
                ResBlock(in_channels=64, out_channels=128, stride=2, downsample=True),
                ResBlock(in_channels=128, out_channels=128, stride=1, downsample=False),
                ResBlock(in_channels=128, out_channels=128, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
                ResBlock(in_channels=128, out_channels=128, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
        )
        
        self.conv4_x = nn.Sequential(
                ResBlock(in_channels=128, out_channels=256, stride=2, downsample=True),
                ResBlock(in_channels=256, out_channels=256, stride=1, downsample=False),
                ResBlock(in_channels=256, out_channels=256, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
                ResBlock(in_channels=256, out_channels=256, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
                ResBlock(in_channels=256, out_channels=256, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
                ResBlock(in_channels=256, out_channels=256, stride=1, downsample=False) 
                        if self.res34 else self.same_layer,
        )

        self.conv5_x = nn.Sequential(
                ResBlock(in_channels=256, out_channels=512, stride=2, downsample=True),
                ResBlock(in_channels=512, out_channels=512, stride=1, downsample=False),
                ResBlock(in_channels=512, out_channels=512, stride=1, downsample=False) 
                        if self.res34 else self.same_layer
        
        ) 
               
        

        
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512,self.output_dim)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pooling(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x