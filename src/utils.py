from PIL import Image
from PIL


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


def count_parameters(model) -> int:
    """this function takes torch model instance and re

    Args:
        model (torch model): torch model instance

    Returns:
        int: the total number of parameters in the input model
    """    

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


