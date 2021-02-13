# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:42:09 2021

@author: Frederik
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from typing import List, Any
import math 
from torch.nn.functional import softplus
from torch.distributions import Distribution
import numpy as np
import torch.optim as optim

from para_fun import sim_plot_3d_fun
import torchvision.transforms as transforms


#Sources:
#https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
#https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb


class vae_para_test(nn.Module):
    
    def __init__(self):
        super(vae_para_test, self).__init__()
        
        self.fch1 = nn.Linear(3, 100)
        
        self.fcmu = nn.Linear(100, 2)
        self.fcvar = nn.Linear(100,2)
        
        self.fcg1 = nn.Linear(2,100)
        self.fcg2 = nn.Linear(100,3)
        #self.device = device
    
    def encoder(self, x):
    
        x = F.relu(self.fch1(x))
        mu = self.fcmu(x)
        var = torch.sigmoid(self.fcvar(x))
        
        return mu, var
        
    def rep_par(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
    def decoder(self, z):
        
        z = F.relu(self.fcg1(z))
        z = self.fcg2(z)
        
        return z
        
    def forward(self, x):
        
        mu, var = self.encoder(x)
        z = self.rep_par(mu, var)
        x_rec = self.decoder(z)
        
        return x_rec, mu, var
        


#Simple test
#vae = vae_para_test()
#x_test = torch.tensor([1,2,3], dtype=torch.torch.float)
#x_rec, _, _ = vae.forward(x_test)

file_path_name = 'para_data.csv'
data_obj = sim_plot_3d_fun(N_sim=50000, name_path = file_path_name)
generate_data = False

if generate_data:
    data_obj.sim_3d()
    
dat = data_obj.read_data()
dat = torch.transpose(dat, 0, 1)

epochs = 10 #100000
batch_size = 100
lr = 0.0001

trainloader = torch.utils.data.DataLoader( dataset = dat , batch_size= batch_size , shuffle = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reconstruction_function = nn.MSELoss(size_average=False)

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

#model = vae_para_test().to(device)
model = vae_para_test()

def loss_fun(x_rec, x, mu, var):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    logvar = torch.log(var)
    BCE = reconstruction_function(x_rec, x) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KL divergence
    return BCE + KLD

optimizer = optim.SGD(model.parameters(), lr=lr)

train_loss = []
val_loss = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x in trainloader:
        #x = x
        #x = x.view(x.size(0), -1)
        #x = Variable(x)
        #x = x.to(device)
        #x.requires_grad=True
        x_rec, mu, var = model(x)
        #bce_loss = criterion(x_rec, data, mu, logvar)
        loss = loss_fun(x_rec, x, mu, var)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        running_loss += loss.item()
        optimizer.step()
        
    train_epoch_loss = running_loss/len(trainloader.dataset)
    train_loss.append(train_epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}: {train_epoch_loss:.4f}")


        
        