import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from src.models.utils import AEDataset
from src.models.utils import count_parameters
from src.models.utils import train_AE, evaluate_AE
from torch.utils.data import DataLoader
from src.models.autoencoder import ResnetAutoencoder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize






data_path = os.path.join(os.curdir,'data','preprocessed', 'autoencoder')


transforms = Compose([
    Resize(224),
    ToTensor(),
    Normalize([0.20041628,0.20041628,0.20041628],
             [0.20288454,0.20288454,0.20288454])
    ])

train_data = AEDataset(data_path,transforms,train =True)
val_data = AEDataset(data_path,transforms,train =False)




model = ResnetAutoencoder(res34=True)


#hyperparametres and setting
lr = 0.0001
batch_size = 4
epochs = 10
weight_decay=0.00001
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
criterion = nn.MSELoss()
schedular = optim.lr_scheduler.StepLR(optimizer, gamma=0.5,step_size=2,verbose=True)
scaler = torch.cuda.amp.GradScaler()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
mu = 0.20041628
sigma = 0.20288454

train_iterator = DataLoader(train_data,shuffle = True, batch_size=batch_size)
val_iterator = DataLoader(train_data,shuffle = True, batch_size=batch_size)

print(f'The model has {count_parameters(model):,} trainable parameters')


best_valid_loss = float('inf')
model_name = 'resnet34_autoencoder_weights.pt'
log = pd.DataFrame(columns=['train_loss' ,'val_loss'])

for epoch in range(epochs):
    
    start_time = time.monotonic()
    
    train_loss = train_AE(model, train_iterator, optimizer, criterion,device,schedular, mu, sigma,scaler=False)
    val_loss = evaluate_AE(model, val_iterator, criterion, device)
        
    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), model_name)

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    log.loc[len(log.index)] = [train_loss,val_loss]
    log.to_csv('log.csv')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s, current time: {time.ctime()}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {val_loss:.3f}')
