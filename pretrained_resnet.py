import os
import time
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



from PIL import Image
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.transforms import Resize, Normalize
from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
from src.models.utils import train, evaluate
from src.models.utils import count_parameters, epoch_time, calculate_accuracy


#random seed setting
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# data directories initiation
train_data_dir = os.path.join(os.curdir,'data','preprocessed','classification','train')
val_data_dir = os.path.join(os.curdir,'data','preprocessed','classification','val')
#ultimate_weights = os.path.join(os.curdir,'exp5','pretrained_resnet34_weights.pt')

#defining the pretrained model
model = resnet34(pretrained=True)

# classification layer defination
INPUT_DIM = model.fc.in_features
OUTPUT_DIM = 4

FC_layer = nn.Linear(INPUT_DIM,OUTPUT_DIM)
model.fc = FC_layer
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

#Weieghts freezing
ct = 0
for child in model.children():
    ct += 1
    if ct <=7:
        for param in child.parameters():
            param.requires_grad = False

print(f'The model has {count_parameters(model):,} trainable parameters')

#hyperparametres and setting
lr = 0.000005
batch_size = 1
epochs = 10
weight_decay=0.00001
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
schedular = optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=1,verbose=True)
scaler = torch.cuda.amp.GradScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# related transformation defination
IMAGE_NET_MEANS = [0.485, 0.456, 0.406]
IMAGE_NET_STDEVS = [0.229, 0.224, 0.225]


transforms = Compose([
                    Resize(224),
                    Lambda(lambda x: x.convert('RGB')),
                    ToTensor(),
                    Normalize(IMAGE_NET_MEANS,IMAGE_NET_STDEVS)
])


# Data loading and labeling
train_data = ImageFolder(root= train_data_dir,
                         transform= transforms,
                         )

val_data = ImageFolder(root= val_data_dir,
                       transform= transforms,
                       )


#data iterator defination

train_iterator = DataLoader(train_data,
                            shuffle = True,
                            batch_size=batch_size)

val_iterator = DataLoader(val_data,
                          shuffle = True,
                          batch_size=batch_size)


criterion = criterion.to(device)
best_valid_loss = float('inf')
model_name = 'pretrained_resnet34_weights.pt'
log = pd.DataFrame(columns=['train_loss','train_acc' ,'val_loss', 'val_acc'])

for epoch in range(epochs):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion,device, schedular,scaler=False)
    val_loss, val_acc = evaluate(model, val_iterator, criterion, device)
        
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), model_name)

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    log.loc[len(log.index)] = [train_loss,train_acc,val_loss,val_acc]
    log.to_csv('log.csv')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s, current time: {time.ctime()}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')