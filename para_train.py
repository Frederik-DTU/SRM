# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:12:32 2021

@author: Frederik
"""

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

#Own files
from VAE_hyper_parabolic import VAE_3d_surface
from para_fun import sim_plot_3d_fun
from riemannian_fun import *

#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

file_path_name = 'para_data.csv'
data_obj = sim_plot_3d_fun(N_sim=50000, name_path = file_path_name)
generate_data = False
try_cuda = False
plot_results = False

if generate_data:
    data_obj.sim_3d()

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
    
dat = data_obj.read_data()

epochs = 10 #100000
batch_size = 100
lr = 0.0001

trainloader = DataLoader( dataset = dat , batch_size= batch_size , shuffle = True)
reconstruction_function = nn.MSELoss(reduction='sum')

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

model = VAE_3d_surface(device = device).to(device)

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
#criterion = nn.BCELoss(reduction='sum')

train_loss = []
val_loss = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x in trainloader:
        #x = x
        #x = x.view(x.size(0), -1)
        #x.requires_grad=True
        x = x.to(device)
        _, x_rec, mu, var = model(x)
        #bce_loss = criterion(x_rec, data, mu, logvar)
        loss = loss_fun(x_rec, x, mu, var)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        running_loss += loss.item()
        optimizer.step()
        
    train_epoch_loss = running_loss/len(trainloader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
    train_loss.append(train_epoch_loss)

torch.save(model.state_dict(), 'test')
model.load_state_dict(torch.load('test'))

learned_dat = model(dat)

T=10 #T in the article (T+1 points)
z0 = torch.Tensor([3,3,0])
zT = torch.Tensor([-3,-3,0])
z0 = model.get_encoded(z0)
zT = model.get_encoded(zT)

riemann = riemannian_dgm(z0, zT, T, 2, 3, model.get_encoded, model.get_decoded)

z_list = riemann.get_zlist()
E_old, E_new, z_path = riemann.geodesic_path_al1(alpha = 0.01, eps = 0.1)    

zl = riemann.get_list_to_torch(z_list, [0,1])
zp = riemann.get_list_to_torch(z_path, [0,1])

z_path = geodesic_path_al1(z_list, model.get_decoded, model.get_encoded,
                            n_out=2, alpha = 1, eps = 0.01)
g0 = model.get_decoded(z0)
get_jacobian(model.get_decoded, z0, 3)

model.get_encoded(z0)

model(torch.Tensor([1,1,1]))