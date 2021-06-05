# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:44:58 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
http://adamlineberry.ai/vae-series/vae-code-experiments
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
"""

#%% Modules

import torch
import torch.optim as optim
import pandas as pd
import numpy as np

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d_prob_var, VAE_3d_prob_logvar, VAE_3d_nprob_var, VAE_3d_nprob_logvar

#%% Function for plotting

def x3_R2(x1, x2):
    
    return x1, x2, 0*x1

def x3_hyper_para(x1, x2):
    
    return x1, x2, x1**2-x2**2

def x3_parabolic(x1, x2):
    
    return x1, x2, x1**2+x2**2

def x3_sphere(x1, x2):
    
    N = len(x1)
    x3 = np.random.normal(0, 1, N)
    
    r = np.sqrt(x1**2+x2**2+x3**2)
    
    return x1/r, x2/r, x3/r

#%% Plotting

data_path = 'Data/parabolic.csv' #'Data/hyper_para.csv'
file_model_save = 'trained_models/parabolic/parabolic_epoch_15000.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
data_plot = plot_3d_fun(N_grid=100, fun = x3_parabolic) #x3_hyper_para
device = 'cpu'
lr = 0.0001

df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Plotting the raw data
x1 = DATA[:,0].detach().numpy()
x2 = DATA[:,1].detach().numpy()
x3 = DATA[:,2].detach().numpy()
data_plot.true_plot_3d([min(x1), max(x1)], [min(x2), max(x2)]) #Plotting the true surface
data_plot.plot_data_scatter_3d(x1, x2, x3) #Plotting the true surface with the simulated data
data_plot.plot_data_surface_3d(x1, x2, x3) #Surface plot of the data

#Plotting the trained model
model = VAE_3d_prob_var().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

X = model(DATA) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
z = X[0]
x_hat = X[1]
mu = X[2]
std = X[3]
x1 = x_hat[:,0].detach().numpy()
x2 = x_hat[:,1].detach().numpy()
x3 = x_hat[:,2].detach().numpy()

#Plotting loss function
data_plot.plot_loss(elbo, title='Loss function')
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')
data_plot.plot_loss(kld_loss, title='KLD')

#Surface plot of the reconstructed data
data_plot.plot_data_surface_3d(x1, x2, x3)

#Plotting the true surface with the reconstructed data
data_plot.plot_data_scatter_3d(x1, x2, x3, title='Scatter of Reconstructed Data')

#Plotting mu in Z-space
z = z.detach().numpy()
mu = mu.detach().numpy()
std = std.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])
data_plot.plot_dat_in_Z_2d([mu, 'mu'])
data_plot.plot_dat_in_Z_2d([std, 'std'])
    
    
    