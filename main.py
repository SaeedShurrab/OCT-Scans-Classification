from src.data.preprocessing import read_iamge, save_image
from src.data.preprocessing import nlm_denoising, resizing
from src.data.preprocessing import create_labels
import os


# Data directories definition
raw_dir = os.path.join(os.curdir,'data','raw','train')
intermed_dir = os.path.join(os.curdir,'data','intermediate')
preprocessed_dir = os.path.join(os.curdir,'data','preprocessed')


# 1- Images denoising process
#for dir in os.listdir(raw_dir):

#    image_dir = os.path.join(raw_dir,dir)

#    saving_dir = os.path.join(intermed_dir,dir)
    
#    for image in os.listdir(image_dir):

#        img = read_iamge(os.path.join(image_dir,image))
#        img = nlm_denoising(img)
#        save_image(img,saving_dir,image,'jpg')


# 2- Data labels extraction
#labesl = [1,2,3,4]
#create_labels(data_path, labesl)


# image resizing
for dir in os.listdir(intermed_dir):

    image_dir = os.path.join(intermed_dir,dir)
    
    saving_dir = os.path.join(preprocessed_dir,dir)
    
    for image in os.listdir(image_dir):
        
        img = read_iamge(os.path.join(image_dir,image))
        img = resizing(img,256,256)
        save_image(img,saving_dir,image, 'jpg')

