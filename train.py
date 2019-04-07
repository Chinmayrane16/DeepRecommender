import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from livelossplot import PlotLosses
from loss import MSEloss_with_Mask
from model import AutoEncoder

def train(model, criterion, optimizer, train_dl, test_dl, num_epochs=40):
    liveloss = PlotLosses()
    for epoch in range(num_epochs):
        train_loss, valid_loss = [], []
        logs = {}
        prefix = ''
  
        # Training Part
        model.train()
        for i, data in enumerate(train_dl, 0):
            # Get the inputs
            inputs = labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            inputs = inputs.float()
            labels = labels.float()
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.cuda()
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            ## -> Dense Output Re-feeding <- ##
            
            # Zero the gradiants
            optimizer.zero_grad()

            # Important detach() the output, to avoid construction of 
            # computation graph
            outputs = model(outputs.detach())
            outputs = outputs.cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            logs[prefix + 'MMSE loss'] = loss.item()
        
        for i, data in enumerate(test_dl, 0):
            model.eval()
            inputs = labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            inputs = inputs.float()
            labels = labels.float()
            
            outputs = model(inputs)
            outputs = outputs.cuda()
            loss = criterion(outputs, labels)
            
            valid_loss.append(loss.item())
            prefix = 'val_'
            logs[prefix + 'MMSE loss'] = loss.item()
        
    print()
    liveloss.update(logs)
    liveloss.draw()
    print ("Epoch:", epoch+1, " Training Loss: ", np.mean(train_loss), " Valid Loss: ", np.mean(valid_loss))