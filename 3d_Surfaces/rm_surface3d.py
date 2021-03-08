# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:55:56 2021

@author: Frederik
"""

#%% Sources


#%% Modules

import torch
import torch.optim as optim
import pandas as pd
import numpy as np

#Own files
from plot_dat import plot_3d_fun
from VAE_surface3d import VAE_3d
from rm_com_v2 import riemannian_dgm

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

#%% Loading module

data_path = 'Data/parabolic.csv' #'Data/hyper_para.csv'
file_model_save = 'trained_models/parabolic/parabolic_epoch_1000.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
data_plot = plot_3d_fun(N_grid=100, fun = x3_parabolic) #x3_hyper_para
device = 'cpu'
lr = 0.0001

df = pd.read_csv(data_path, index_col=0)
DATA = torch.Tensor(df.values)
DATA = torch.transpose(DATA, 0, 1)

#Loading the model
model = VAE_3d().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

#%% Finding geodesic between two points

T = 10
rm = riemannian_dgm(T, 3, 2, model.h, model.g)
test = torch.Tensor([1,1,2])
print(model(test.float()))

x = torch.Tensor([-3,-3,0])
y = torch.Tensor([3,-3,0])
hx, _, _ = model.h(x)
hy, _, _ = model.h(y)


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
    
    