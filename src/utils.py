import os
from typing import Any


from PIL import Image
from typing import Tuple



def read_iamge(img_dir: str) -> Any:
    """this function read an image from the supplied path

    Args:
        img_dir (str): path to the input image

    Returns:
        PIL Image
    """                
    img = Image.open(img_dir)
    return img


def save_image(img:Any, output_dir: str, img_name: str ):
    """this function convert the input image into uint8 class and save it to a spicific directory

    Args:
        img (Any): input image of type
        output_dir (str): the location at which the image will be saved
        img_name (str): the ultimate name of the image in the form image.format 
    """    
    img.save(fp=os.path.join(output_dir,img_name))










