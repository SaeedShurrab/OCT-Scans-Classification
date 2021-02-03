import numpy as np
import pandas as pd
from typing import List, Any
import os

import skimage
from PIL import Image
from skimage.restoration import  denoise_nl_means
from skimage.metrics import peak_signal_noise_ratio
import skimage.io as io


import torchvision.transforms.functional as ft



def nlm_denoising(img:Any) -> Any:
    """ this function takes noisy image and return a denoised version of the input image 

    Args:
        img (Any): input image

    Returns:
        output image
    """    
    img= np.array(img)   
    img = skimage.img_as_float(img)
    img = denoise_nl_means(img,patch_size=7,patch_distance=11,h=0.1)
    img = skimage.img_as_ubyte(img)
    img = Image.fromarray(img)
    return img




def resizing(img:Any , H:int, W:int) -> Any:
    """this function takse input image of any extension and return a resized version 
    of it according the specified width and hight

    Args:
        img (Any): the input image
        H (int): the hight of the resized image
        W (int): the width of the resized image

    Returns:
        the resized image
    """    
    img = ft.resize(img,size=[H,W])
    return img



def hflip_aug(img:Any) -> Any:
    """ this function takes input image, perform horizontal flipping and return it


    Args:
        img (Any): the input image

    Returns:
        Any: the output image
    """
    img = ft.hflip(img)    
    return img



def randrot_aug(img: Any, direction: str = 'positive' ) -> Any:
    """this function takes input image, perform random rotation and return it.

    Args:
        img (Any): [description]
        direction (str, optional): the rotation direction. Defaults to 'positive'.

    Returns:
        the reotated image
    """    
    angle = np.random.randint(8,16)

    if direction == 'negative':
        angle *= -1

    img = ft.rotate(img,angle,fill=255)
    return img


def randshear_aug(img: Any, direction: str = 'positive' ) -> Any:
    """this function takes input image, perform random shearing and return it.

    Args:
        img (Any): [description]
        direction (str, optional): the shearing direction. Defaults to 'positive'.

    Returns:
        the sheared image
    """    
    angle = np.random.randint(5,30)

    if direction == 'negative':
        angle *= -1

    img = ft.affine(img,angle=0,translate=[0,0],scale=1,shear=angle,fillcolor= 255)
    return img



