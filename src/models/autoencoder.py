import os
import torch
import numpy as np


from torch import nn
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Dataset,DataLoader



class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
                nn.Conv2d(**kwargs),
                nn.BatchNorm2d(num_features=kwargs['out_channels']),       
        )
        
    def forward(self,x):
        return self.block(x)



class ConvTransBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvTransBlock, self).__init__()
        
        self.block = nn.Sequential(
                nn.ConvTranspose2d(**kwargs),
                nn.BatchNorm2d(num_features=kwargs['out_channels']),
                nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        return self.block(x)


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


class ResnetAutoencoder(nn.Module):
    def __init__(self,res34=False):
        super(ResnetAutoencoder,self).__init__()
        
        self.res34 = res34
        self.same_layer = nn.MaxPool2d(kernel_size=1,stride=1)
        
        
        #Encoder
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
               
        
        #Decoder
        self.block6 = nn.Sequential(
                
                #ConvTransBlock(in_channels= 512,out_channels=512, kernel_size=3, padding=1),
                ConvTransBlock(in_channels= 512, out_channels = 256, kernel_size= 2, stride=2),
            
                #ConvTransBlock(in_channels= 256,out_channels=256, kernel_size=3, padding=1),
                ConvTransBlock(in_channels= 256, out_channels = 128, kernel_size= 2, stride=2),
                
                #ConvTransBlock(in_channels= 128,out_channels=128, kernel_size=3, padding=1),
                ConvTransBlock(in_channels= 128, out_channels = 64,  kernel_size= 2, stride=2),
            
                #ConvTransBlock(in_channels= 64,out_channels=64, kernel_size=3, padding=1),
                ConvTransBlock(in_channels= 64, out_channels = 64,   kernel_size= 2, stride=2),
                
                ConvTransBlock(in_channels= 64,out_channels=64, kernel_size=3, padding=1),
                ConvTransBlock(in_channels= 64, out_channels = 3,    kernel_size= 2, stride=2),
        
        
        )
        
        #self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        #self.fc = nn.Linear(512,1000)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.block6(x)
        return x



def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    
    model.train()
    
    for images in iterator:
        
        images = images.to(device)
        
        optimizer.zero_grad()
                
        output = model(images)
        
        loss = criterion(output, images)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for images in iterator:
            
            images = images.to(device)
            
            output = model(images)

            loss = criterion(output, images)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)