
import os
import torch

from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm


def count_parameters(model) -> int:
    """this function takes torch model instance and re

    Args:
        model (torch model): torch model instance

    Returns:
        int: the total number of parameters in the input model
    """    

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time: float, end_time: float) -> Tuple[int,int]:
    """ this function calculate the time per epoch

    Args:
        start_time (float): epoch starting time
        end_time (float): epoch ending time

    Returns:
        Tuple[int,int]: elapsed time in minute, elapsed time in second
    """        

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#accuracy calculation function
def calculate_accuracy(y_pred, y) -> float:
    """This function is responsible for accuracy calculation

    Args:
        y_pred : predicted labels tensor
        y : true labels tensor

    Returns:
        float: accuracy score
    """    
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


class AEDataset(Dataset):
    def __init__(self, data_path, transforms,train=True ):
        if train :
            self.data_path = os.path.join(data_path, 'train')
        else:
            self.data_path = os.path.join(data_path, 'val')
        self.all_images = os.listdir(self.data_path)
        self.trasforms = transforms
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx)->torch.tensor:
        image_name = self.all_images[idx]
        path = os.path.join(self.data_path, image_name)
        return self.trasforms(Image.open(fp=path).convert('RGB'))


def train_AE(model, iterator, optimizer, criterion, device,schedular ,mu, sigma, scaler= None) -> float:
    """ This function is responsible for performing the training loop and returning the loss 
    training loss value for each epoch for the autoencoder model

    Args:
        model : pytorch model instance
        iterator : train data iterator
        optimizer : the specified optimization method
        criterion : the specified loss function 
        device : the divice at which the compuation will be performed 
        schedular : the specified learning rate schedular
        mu: train data global mean
        sigma: train data global standard deviation
        scaler : computation scaler for Automatic mixed precision enabling, pass the scaler instance .

    Returns:
        float: training loss per epoch
    """    

    print('training')
    
    epoch_loss = 0
    
    model.train()
    
    for image in tqdm(iterator):
        
        image = image.to(device)
        
        optimizer.zero_grad()
        
        # Automatic mixed precision option
        if scaler:
            
            with torch.cuda.amp.autocast():     
                
                output = model(image)
                image = (image * sigma) + mu 
                loss = criterion(image, output)
                assert output.dtype is torch.float16
        
        
        #float32 precision option         
        else:
            output = model(image)
            image = (image * sigma) + mu 
            loss = criterion(image, output)
        
        
        epoch_loss += loss.item()

        # Automatic mixed precision option
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        #float32 precision option 
        else:
            loss.backward()
            optimizer.step()
        
    schedular.step()

        
    return epoch_loss / len(iterator)


def evaluate_AE(model, iterator, criterion, device) -> float:
    """ This function is responsible for performing the validation loop and returning the loss 
    training loss value for each epoch for the autoencoder model

    Args:
        model : pytorch model instance
        iterator : train data iterator
        optimizer : the specified optimization method
        criterion : the specified loss function 
        device : the divice at which the compuation will be performed 


    Returns:
        float: validation loss per epoch
    """   
    
    
    print('validating')
    epoch_loss = 0

    
    model.eval()
    
    with torch.no_grad():
        
        for image in tqdm(iterator):

            image = image.to(device)

            output = model(image)

            image = (image * sigma) + mu 

            loss = criterion(image, output)

            epoch_loss += loss.item()

        
    return epoch_loss / len(iterator)



#training loop defination
def train(model, iterator, optimizer, criterion, device,schedular ,scaler= False) -> float:
    """ This function is responsible for performing the training loop and returning the loss 
    training loss value for each epoch for the classification model

    Args:
        model : pytorch model instance
        iterator : train data iterator
        optimizer : the specified optimization method
        criterion : the specified loss function 
        device : the divice at which the compuation will be performed 
        schedular : the specified learning rate schedular
        scaler : computation scaler for Automatic mixed precision enabling, pass the scaler instance .

    Returns:
        float: training loss per epoch
    """    
    print('training')
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for (image, label) in tqdm(iterator):
        
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        #mixed precision option
        if scaler:
            
            with torch.cuda.amp.autocast():     
                
                label_pred = model(image)
                loss = criterion(label_pred, label)
                assert label_pred.dtype is torch.float16
                
        else:
            label_pred = model(image)
            loss = criterion(label_pred, label)
        
        acc = calculate_accuracy(label_pred, label)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        else:
            loss.backward()
            optimizer.step()
        
    schedular.step()

        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """ This function is responsible for performing the validation loop and returning the loss 
    training loss value for each epoch for the classification model

    Args:
        model : pytorch model instance
        iterator : train data iterator
        optimizer : the specified optimization method
        criterion : the specified loss function 
        device : the divice at which the compuation will be performed 


    Returns:
        float: validation loss per epoch
    """  
    print('validating')
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (image, label) in tqdm(iterator):

            image = image.to(device)
            label = label.to(device)

            label_pred = model(image)

            loss = criterion(label_pred, label)

            acc = calculate_accuracy(label_pred, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)