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
from VAE_test import vae_para_test

#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
#https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder

#%% Training

#def init_weights(m):
#    if type(m) == nn.Linear:
#        torch.nn.init.xavier_uniform(m.weight)
#        m.bias.data.fill_(0.01)

test_model = False
file_path_name = 'Data/para_data.csv'
file_model_save = 'trained_models/para_3d.pt'
data_obj = sim_3d_fun(N_sim=50000, name_path = file_path_name)
generate_data = False
try_cuda = False
save_step = 1000
beta = 1
alpha = 3

epochs = 1000 #100000
if epochs==0:
    epoch=0
batch_size = 100
lr = 0.0001

if generate_data:
    data_obj.sim_3d()

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

dat = (data_obj.read_data()).to(device)

batch_size = 100 #100
if try_cuda:
    trainloader = DataLoader(dataset = dat , batch_size= batch_size , shuffle = True,
                             num_workers = 4)
else:
    trainloader = DataLoader(dataset = dat , batch_size= batch_size , shuffle = True)

rec_fun = nn.MSELoss(reduction='mean')

if test_model:
    model = vae_para_test(device = device).to(device)
else:
    model = VAE_3d_surface(device = device).to(device)

#model.apply(init_weights)
def loss_fun(x_rec, x, mu, var):
    """
    This function will add the reconstruction loss (MSELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    logvar = torch.log(var)
    BCE = rec_fun(x_rec, x)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - var, dim = 1), dim = 0)
    # KL divergence
    return alpha*BCE, beta*KLD, alpha*BCE + beta*KLD

optimizer = optim.SGD(model.parameters(), lr=lr)

train_loss_tot = []
train_loss_bce = []
train_loss_kld = []
for epoch in range(epochs):
    model.train()
    running_loss_tot = 0.0
    running_loss_bce = 0.0
    running_loss_kld = 0.0
    for x in trainloader:
        x = x.to(device)
        _, x_rec, mu, var = model(x)
        loss = loss_fun(x_rec, x, mu, var)
        running_loss_bce += loss[0]
        running_loss_kld += loss[1]
        loss = loss[2]
        optimizer.zero_grad()
        loss.backward()
        running_loss_tot += loss.item()
        optimizer.step()

    train_epoch_loss = running_loss_tot/len(trainloader.dataset)
    train_loss_tot.append(train_epoch_loss)
    train_loss_bce.append(running_loss_bce/len(trainloader.dataset))
    train_loss_kld.append(running_loss_kld/len(trainloader.dataset))
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
    if epoch % save_step == 0:
        torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_tot,
            'BCE_loss': train_loss_bce,
            'KLD_loss': train_loss_kld
            }, file_model_save)


torch.save({'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_tot,
            'BCE_loss': train_loss_bce,
            'KLD_loss': train_loss_kld
            }, file_model_save)