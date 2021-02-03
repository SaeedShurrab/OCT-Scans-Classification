from tqdm import tqdm
from src.utils import read_iamge, save_image
from src.data.preprocessing import nlm_denoising, resizing
from src.data.preprocessing import hflip_aug, randrot_aug , randshear_aug

import os

from torchvision.datasets import ImageFolder
import torch




# Data directories definition
raw_dir = os.path.join(os.curdir,'data','raw','train')
intermed_dir = os.path.join(os.curdir,'data','intermediate','train')
preprocessed_dir = os.path.join(os.curdir,'data','preprocessed','test')


# 1- Images denoising process
#for dir in os.listdir(raw_dir):

#    image_dir = os.path.join(raw_dir,dir)

#    saving_dir = os.path.join(intermed_dir,dir)
    
#    for image in tqdm(os.listdir(image_dir)):

#        img = read_iamge(os.path.join(image_dir,image))
#        img = nlm_denoising(img)
#        save_image(img,saving_dir,image)




# 2- image resizing
for dir in os.listdir(raw_dir):

    image_dir = os.path.join(raw_dir,dir)
    
    saving_dir = os.path.join(intermed_dir,dir)
    
    for image in tqdm(os.listdir(image_dir)):
        
        img = read_iamge(os.path.join(image_dir,image))
        img = resizing(img,224,224)
        save_image(img,saving_dir,image)



# 3-images augmentation
#augmentations = ['h-flip','pos_rot', 'neg_rot', 'pos_shear', 'neg_shear']

#for augment in augmentations:
#    i = 0
#    for dir in os.listdir(intermed_dir):

#        if dir in ['CNV']:

#            image_dir = os.path.join(intermed_dir,dir)
    
#            saving_dir = os.path.join(preprocessed_dir,dir)
            
#            for image in tqdm(os.listdir(image_dir)):

#                img = read_iamge(os.path.join(image_dir,image))

#                if augment == 'h-flip':
#                    i+=1
#                    img = hflip_aug(img)
#                    image = image[:-5] + '-hf' + image[-5:]
#                    if i == 281:
#                        break
                
#                if augment == 'pos_rot':
#                    i+=1
#                    img = randrot_aug(img, direction='positive')
#                    image = image[:-5] + '-pr' + image[-5:]
#                    if i == 281:
#                        break

#                if augment == 'neg_rot':
#                    i+=1
#                    img = randrot_aug(img,direction= 'negative')
#                    image = image[:-5] + '-nr' + image[-5:]
#                    if i == 281:
#                        break

#                if augment == 'pos_shear':
#                    i+=1
#                    img = randshear_aug(img,direction= 'positive')
#                    image = image[:-5] + '-psh' + image[-5:]
#                    if i == 281:
#                        break
                
#                if augment == 'neg_shear':
#                    i+=1
#                    img = randshear_aug(img,direction= 'negative')
#                    image = image[:-5] + '-nsh' + image[-5:]
#                    if i == 281:
#                        break

#                img = resizing(img,224,224)
#                save_image(img,saving_dir,image)







