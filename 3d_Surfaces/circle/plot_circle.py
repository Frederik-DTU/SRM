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
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch import nn
import math
pi = math.pi

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d

#%% Function for plotting

def circle_fun(N = 1000, mu = np.array([1.,1.,1.]), r=1):
    
    theta = np.linspace(0, 2*np.pi, N)
    x1 = r*np.cos(theta)+mu[0]
    x2 = r*np.sin(theta)+mu[1]
    x3 = np.zeros(N)+mu[2]
    
    return x1, x2, x3

#%% Loading data and model

#Hyper-parameters
epoch_load = '40000'
lr = 0.0001
device = 'cpu'
latent_dim = 2

#Data
data_name = 'circle'
fun = circle_fun

#Loading files
data_path = 'Data/'+data_name+'.csv'
file_model_save = 'trained_models/latent_dim_2/'+data_name+'_epoch_'+epoch_load+'.pt'
data_plot = plot_3d_fun(N=100)

#Loading data
df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading model
model = VAE_3d(fc_h = [3, 100],
                 fc_g = [latent_dim, 100, 3],
                 fc_mu = [100, latent_dim],
                 fc_var = [100, latent_dim],
                 fc_h_act = [nn.ELU],
                 fc_g_act = [nn.ELU, nn.Identity],
                 fc_mu_act = [nn.Identity],
                 fc_var_act = [nn.Sigmoid]).to(device) #Model used
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
X = DATA.detach().numpy()
data_plot.true_path3d_with_points(fun, X, [0, 2], [0,2], [0,2]) #Plotting the true surface

#%% Plotting learned data

X = model(DATA) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
z = X[0]
x_hat = X[1].detach().numpy()
mu = X[2]
std = X[3]

#Plotting loss function
data_plot.plot_loss(elbo, title='Loss function')
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')
data_plot.plot_loss(kld_loss, title='KLD')

data_plot.true_path3d_with_points(fun, x_hat, [0, 2], [0,2], [0,2]) #Plotting the true surface
#Plotting mu in Z-space

z = z.detach().numpy()
mu = mu.detach().numpy()
std = std.detach().numpy()

data_plot.plot_1d_hist(z, 'Latent Dimension')
data_plot.plot_1d_hist(mu, 'Mean Value')
data_plot.plot_1d_hist(std, 'Standard Deviation')

#%% Plotting the Riemannian simple geodesics

load_path = 'rm_computations/simple_geodesic_2/'
names = ['pi_2', 'pi', '5pi_4']

for i in range(len(names)):
    
    load = load_path+names[i]+'.pt'
    checkpoint = torch.load(load, map_location=device)
    gx = checkpoint['gx']
    gy = checkpoint['gy']
    points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)
    
    G_old = checkpoint['G_old'].detach().numpy()
    G_new = checkpoint['G_new'].detach().numpy()
    L_old = checkpoint['L_old']
    L_new = checkpoint['L_new']
    
    data_plot.plot_geodesic3d(circle_fun, points.detach().numpy(),
                              [0,2],[0,2],[0,2],
                              [G_old, 'Interpolation (L=%.4f)'%L_old], 
                              [G_new, 'Approximated Geodesic (L=%.4f)'%L_new])

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

data_plot.plot_means_with_true_shape3d(circle_fun, data_batch,
                              [mug_init.detach().numpy(), 'Linear mean'], 
                              [mu_g.detach().numpy(), 'Approximated Frechét mean'])






    