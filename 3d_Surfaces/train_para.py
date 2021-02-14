# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:12:32 2021

@author: Frederik
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

#Own files
from VAE_3d import VAE_3d_surface
from sim_dat import sim_3d_fun

#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

#%% Training

file_path_name = 'Data/para_data.csv'
file_model_save = 'trained_models/para_3d.pt'
data_obj = sim_3d_fun(N_sim=50000, name_path = file_path_name)
generate_data = False
try_cuda = False
save_step = 100

epochs = 100000
batch_size = 100
lr = 0.0001#0.0001

if generate_data:
    data_obj.sim_3d()

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

dat = data_obj.read_data()

#trainloader = DataLoader( dataset = dat , batch_size= batch_size , shuffle = False)
trainloader = DataLoader( dataset = dat , batch_size= batch_size , shuffle = True)
reconstruction_function = nn.MSELoss(reduction='sum')

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

train_loss = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x in trainloader:
        x = x.to(device)
        _, x_rec, mu, var = model(x)
        loss = loss_fun(x_rec, x, mu, var)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        running_loss += loss.item()
        optimizer.step()

    train_epoch_loss = running_loss/len(trainloader.dataset)
    #print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
    train_loss.append(train_epoch_loss)
    if epoch % save_step == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, file_model_save)


torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
            }, file_model_save)
