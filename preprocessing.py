from src.utils import read_iamge, save_image
from src.data.preprocessing import nlm_denoising, resizing
from src.data.preprocessing import hflip_aug, randrot_aug

import os

from torchvision.datasets import ImageFolder
import torch




# Data directories definition
raw_dir = os.path.join(os.curdir,'data','raw','train')
intermed_dir = os.path.join(os.curdir,'data','intermediate')
preprocessed_dir = os.path.join(os.curdir,'data','preprocessed')


# 1- Images denoising process
for dir in os.listdir(raw_dir):

    image_dir = os.path.join(raw_dir,dir)

    saving_dir = os.path.join(intermed_dir,dir)
    
    for image in os.listdir(image_dir):

        img = read_iamge(os.path.join(image_dir,image))
        img = nlm_denoising(img)
        save_image(img,saving_dir,image,'jpg')




# 2- image resizing
for dir in os.listdir(intermed_dir):

    image_dir = os.path.join(intermed_dir,dir)
    
    saving_dir = os.path.join(preprocessed_dir,dir)
    
    for image in os.listdir(image_dir):
        
        img = read_iamge(os.path.join(image_dir,image))
        img = resizing(img,256,256)
        save_image(img,saving_dir,image)



# 3-images augmentation
augmentations = ['h-flip','pos_rot', 'neg_rot']

for augment in augmentations:

    for dir in os.listdir(intermed_dir):

        if dir in ['DME','DRUSEN']:

            image_dir = os.path.join(intermed_dir,dir)
    
            saving_dir = os.path.join(preprocessed_dir,dir)
    
            for image in os.listdir(image_dir):
        
                img = read_iamge(os.path.join(image_dir,image))

                if augment == 'h-flip':
                    img = hflip_aug(img)
                    image = image[:-5] + '-hf' + image[-5:]
                
                if augment == 'pos_rot':
                    img = randrot_aug(img, direction='positive')
                    image = image[:-5] + '-pr' + image[-5:]

                if augment == 'neg_rot':
                    img = randrot_aug(img,direction= 'negative')
                    image = image[:-5] + '-nr' + image[-5:]
                

                img = resizing(img,256,256)
                save_image(img,saving_dir,image)







