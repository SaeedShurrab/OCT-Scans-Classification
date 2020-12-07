import numpy as np
import pandas as pd
from typing import List, Any
import os

import skimage
from skimage.restoration import  denoise_nl_means
from skimage.metrics import peak_signal_noise_ratio
import skimage.io as io

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as ft

def create_labels(data_path:str, labels: List): 
    """this function takes a the path at which the data is located along with list of numerical
    labels and return a csv file that contain the image name along with the class it belongs to 

    Arguments:
        data_path {str} -- the directory at which the data is located 
        labels {List} -- the numerical mapping of each class labels
    """    

    df = pd.DataFrame(columns=['name','label'])

    for idx,dir in enumerate(os.listdir(data_path)):

        temp_df = pd.DataFrame(columns=['name','label'])
        temp_df['name'] = os.listdir(os.path.join(data_path,dir))
        temp_df['label'] = labels[idx]
        df = pd.concat([df,temp_df],ignore_index=True)
    
    df.to_csv(os.path.join(data_path, 'labels.csv'))



def read_iamge(img_dir: str) -> Any:
    """ this function read an image store it as numpy array convert it into image of type (float)

    Arguments:
        img_dir {str} -- path to the input image

    Returns:
        image
    """    
    img = Image.open(img_dir)
    return img


def nlm_denoising(img:Any) -> np.ndarray:
    """this function takes noisy image and return a denoised version of the input image 

    Arguments:
        img {np.ndarray} -- PIL inputt image

    Returns:
        np.ndarray -- denoised image as numpy arra
    """ 
    img= np.array(img)   
    img = skimage.img_as_float(img)
    nlm_den = denoise_nl_means(img,patch_size=7,patch_distance=11,h=0.1)
    img = skimage.img_as_ubyte(img)
    img = Image.fromarray(img)
    return nlm_den

def save_image(img:Any, output_dir: str, img_name: str, extension:str ):
    """ this function convert the input image into uint8 class and save it to a spicific directory

    Arguments:
        img {Any} -- the image to be saved
        output_dir {str} -- image location path
        img_name {str} -- the name of the denoised image
        extension {str} -- 
    """    
    img.save(fp=os.path.join(output_dir,img_name + '.' + extension))


def hflip_aug(img:np.ndarray) -> np.ndarray:
    """this function takes input image, perform horizontal flipping and resizing
    and return it as uint-8 

    Arguments:
        img {np.ndarray} -- input image

    Returns:
        np.ndarray -- h-flipped image
    """  
    
    img = ft.hflip(img)
    img = ft.resize(img,size=[256, 256])
    
    return img

def resizing(img:Any , H:int, W:int) -> Any:
    """this function takse input image of any extension and return a resized version 
    of it according the specified width and hight

    Arguments:
        img {Any} -- the image to be resized
        H {int} -- the hight of the resized image
        W {int} -- the width of the resized image

    Returns:
        Any -- [description]
    """
    img = ft.resize(img,size=[H,W])
    return img


   

class RoctDataset(Dataset):

    def __init__(self, data_path:str, labels, transform= None):
        self.data_path = data_path
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pass
