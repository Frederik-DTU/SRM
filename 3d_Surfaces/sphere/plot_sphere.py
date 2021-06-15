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

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d

#%% Function for plotting

def x3_sphere(x1, x2):
    
    N = len(x1)
    x3 = np.random.normal(0, 1, N)
    
    r = np.sqrt(x1**2+x2**2+x3**2)
    
    return x1/r, x2/r, x3/r

#%% Loading data and model

#Hyper-parameters
epoch_load = '100000'
lr = 0.0001
device = 'cpu'

#Parabolic data
data_name = 'sphere'
fun = x3_sphere

#Loading files
data_path = 'Data/'+data_name+'.csv'
file_model_save = 'trained_models/main2/'+data_name+'_epoch_'+epoch_load+'.pt'
data_plot = plot_3d_fun(N=100)

#Loading data
df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading model
model = model = VAE_3d().to(device) #Model used
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

#%% Plotting true data

#Plotting the raw data
x1 = DATA[:,0].detach().numpy()
x2 = DATA[:,1].detach().numpy()
x3 = DATA[:,2].detach().numpy()
data_plot.true_Surface3d(fun, [min(x1), max(x1)], [min(x2), max(x2)]) #Plotting the true surface
data_plot.plot_data_scatter_3d(fun, x1, x2, x3) #Plotting the true surface with the simulated data
data_plot.plot_data_surface_3d(x1, x2, x3) #Surface plot of the data

#%% Plotting learned data

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
data_plot.plot_data_scatter_3d(fun, x1, x2, x3, title='Scatter of Reconstructed Data')

#Plotting mu in Z-space
z = z.detach().numpy()
mu = mu.detach().numpy()
std = std.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])
data_plot.plot_dat_in_Z_2d([mu, 'mu'])
data_plot.plot_dat_in_Z_2d([std, 'std'])

#%% Plotting the Riemannian simple geodesics

load_path = 'rm_computations/simple_geodesic/simple_geodesic.pt'

checkpoint = torch.load(load_path, map_location=device)
gx = checkpoint['gx']
gy = checkpoint['gy']
points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)

Z_old = checkpoint['Z_old'].detach().numpy()
Z_new = checkpoint['Z_new'].detach().numpy()
G_old = checkpoint['G_old'].detach().numpy()
G_new = checkpoint['G_new'].detach().numpy()
L_old = checkpoint['L_old']
L_new = checkpoint['L_new']

data_plot.plot_geodesic_in_X_3d(fun, #points.detach().numpy(),
                          [-4,4],[-4,4],
                          [G_old, 'Interpolation (L=%.4f)'%L_old], 
                          [G_new, 'Approximated Geodesic (L=%.4f)'%L_new])

data_plot.plot_geodesic_in_Z_2d([Z_old, 'Interpolation (L=%.4f)'%L_old], 
                          [Z_new, 'Approximated Geodesic (L=%.4f)'%L_new])

#%% Plotting Frechet mean

load = 'rm_computations/frechet_mean/frechet_mean.pt'
checkpoint = torch.load(load, map_location=device)
L = checkpoint['L']
muz_init = checkpoint['muz_init'].view(-1)
mug_init = checkpoint['mug_init'].view(-1)
mu_z = checkpoint['mu_z'].view(-1)
mu_g = checkpoint['mu_g'].view(-1)
batch_size = checkpoint['batch_size']

data_batch = DATA[0:batch_size].detach().numpy()
Z = model.h(DATA[0:batch_size])[0]

data_plot.plot_means_with_true_surface3d(fun, data_batch, [-2,2], [-2,2],
                              [-4,4],[-4,4],[-4,4],
                              [mug_init.detach().numpy(), 'Linear mean'], 
                              [mu_g.detach().numpy(), 'Approximated Frechét mean'])

data_plot.plot_mean_in_Z2d(Z.detach().numpy(), [muz_init.detach().numpy(), 'Linear mean'], 
                           [mu_z.detach().numpy(), 'Approximated Frechét mean'])
    
    
    
    