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
import sympy as sym

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d
from rm_computations import rm_geometry

#%% Function for plotting

def fun(x1, x2):
    
    return x1, x2, x1**2+x2**2 

x1, x2 = sym.symbols('x1 x2')
x = sym.Matrix([x1, x2])
param_fun = sym.Matrix([x1, x2, x1**2+x2**2])

#Data
data_name = 'paraboloid'

#%% Loading data and model

#Hyper-parameters
epoch_load = '100000'
lr = 0.0001
device = 'cpu'

#Loading files
data_path = 'Data/'+data_name+'.csv'
file_model_save = 'trained_models/'+data_name+'/'+data_name+'_epoch_'+epoch_load+'.pt'
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

#Class to the Riemannian Geometry assumining known metric matrix function
rm = rm_geometry()
rm.compute_mmf(param_fun, x)

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

load_path = 'rm_computations/'+data_name+'/simple_geodesic.pt'

checkpoint = torch.load(load_path, map_location=device)
g_geodesic = checkpoint['g_geodesic']
gx = g_geodesic[0]
gy = g_geodesic[-1]
points = torch.transpose(torch.cat((gx.view(-1,1), gy.view(-1,1)), dim = 1), 0, 1)

z_linear = checkpoint['z_linear'].detach().numpy()
z_geodesic = checkpoint['z_geodesic'].detach().numpy()
g_linear = checkpoint['g_linear'].detach().numpy()
g_geodesic = g_geodesic.detach().numpy()
L_linear = checkpoint['L_linear']
L_geodesic = checkpoint['L_geodesic']

zx = np.array([-3.0,-3.0])
zy = np.array([3.0,-3.0])
y_init = np.zeros((4, 100))
z_true_geodesic = rm.bvp_geodesic(zx, zy, 100, y_init).transpose()
g1, g2, g3 = fun(z_true_geodesic[:,0], z_true_geodesic[:,1])
g_true_geodesic = np.vstack((g1,g2,g3)).transpose()
L_true = rm.arc_length(g_true_geodesic)

data_plot.plot_geodesic_in_X_3d(fun, #points.detach().numpy(),
                          [-4,4],[-4,4],
                          [g_linear, 'Interpolation (L=%.4f)'%L_linear], 
                          [g_geodesic, 'Approximated Geodesic (L=%.4f)'%L_geodesic],
                          [g_true_geodesic, 'True Geodesic (L=%.4f)'%L_true])

data_plot.plot_geodesic_in_Z_2d([z_linear, 'Interpolation (L=%.4f)'%L_linear], 
                          [z_geodesic, 'Approximated Geodesic (L=%.4f)'%L_geodesic])

#%% Plotting Frechet mean

load_path = 'rm_computations/'+data_name+'/frechet_mean.pt'
checkpoint = torch.load(load_path, map_location=device)
loss = checkpoint['loss']
muz_linear = checkpoint['muz_linear'].view(-1)
mug_linear = checkpoint['mug_linear'].view(-1)
muz_geodesic = checkpoint['muz_geodesic'].view(-1)
mug_geodesic = checkpoint['mug_geodesic'].view(-1)
batch_size = checkpoint['batch_size']

data_batch = DATA[0:batch_size].detach().numpy()
Z = model.h(DATA[0:batch_size])

X = np.zeros((list(Z.shape)))
X[:,0] = data_batch[:,0]
X[:,1] = data_batch[:,1]
true_mean = rm.karcher_mean_algo(X.transpose())
true_mean = true_mean[:,-1]
g1, g2, g3 = fun(true_mean[0], true_mean[1])
g_true_mean = np.vstack((g1,g2,g3))

data_plot.plot_means_with_true_surface3d(fun, data_batch, [-2,2], [-2,2],
                              [-4,4],[-4,4],[-4,4],
                              [mug_linear.detach().numpy(), 'Linear mean'], 
                              [mug_geodesic.detach().numpy(), 'Approximated Frechét mean'],
                              [g_true_mean, 'True Frechet mean'])

data_plot.plot_mean_in_Z2d(Z.detach().numpy(), [muz_linear.detach().numpy(), 'Linear mean'], 
                           [mug_linear.detach().numpy(), 'Approximated Frechét mean'])
    