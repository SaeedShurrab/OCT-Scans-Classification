import os
import time
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


from PIL import Image
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader 

from src.models.AEresnet import AEResnet
from src.models.utils import train, evaluate
from src.models.utils import count_parameters, calculate_accuracy, epoch_time


from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.transforms import Resize, Normalize
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix


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
weights_path = os.path.join(os.curdir,'models','Autoencoder-weights','resnet34.pt')
#ultimate_weights = os.path.join(os.curdir,'exp10','AEpretrained_resnet34_weights.pt')

#defining the pretrained model
model = AEResnet(res34=True,output_dim=4)

# Auto encoder data loading
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')),strict = False)

# classification layer defination
INPUT_DIM = model.fc.in_features
OUTPUT_DIM = model.fc.out_features

FC_layer = nn.Linear(INPUT_DIM,OUTPUT_DIM)
model.fc = FC_layer
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True

print(f'The model has {count_parameters(model):,} trainable parameters')

#hyperparametres and setting
lr = 0.001
batch_size = 1
epochs = 10
weight_decay=0
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
schedular = optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=1,verbose=True)
scaler = torch.cuda.amp.GradScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# related transformation defination
ROCT_MEANS = [0.20041628,0.20041628,0.20041628]
ROCT_STDEVS = [0.20288454,0.20288454,0.20288454]



transforms = Compose([
                    Resize(224),
                    Lambda(lambda x: x.convert('RGB')),
                    ToTensor(),
                    Normalize(ROCT_MEANS,ROCT_STDEVS)
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


# Model Training loop defination
best_valid_loss = float('inf')
model_name = 'AEpretrained_resnet34_weights.pt'
log = pd.DataFrame(columns=['train_loss','train_acc' ,'val_loss', 'val_acc'])

for epoch in range(epochs):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion,device,schedular,scaler=False)
    val_loss, val_acc = evaluate(model, val_iterator, criterion, device)
        
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), model_name)

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    log.loc[len(log.index)] = [train_loss,train_acc,val_loss,val_acc]
    log.to_csv('log.csv')
    
#     print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s, current time: {time.ctime()}')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#     print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')