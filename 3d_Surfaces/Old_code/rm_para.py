# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 01:24:56 2021

@author: Frederik
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

#Own files
from VAE_3d import VAE_3d_surface
from plot_dat import plot_3d_fun
from sim_dat import sim_3d_fun
from rm_com import riemannian_dgm


#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

#%% Loading

file_path_name = 'Data/para_data.csv'
file_model_save = 'trained_models/para_3d.pt'
data_obj = sim_3d_fun(N_sim=50000, name_path = file_path_name)
data_plot = plot_3d_fun(N_grid=100)
try_cuda = False

epochs = 10#100000
batch_size = 100
lr = 0.0001

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

dat = data_obj.read_data()

model = VAE_3d_surface(device = device).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()

#%% Riemannian Computations

T = 10 #T defined as in the article (corresponds to T+1 points)

x = torch.tensor([-3,-3,0], dtype=torch.float32).to(device) #x as noted in article
y = torch.tensor([3,-3,0], dtype=torch.float32).to(device) #y as noted in article

zx = model.get_encoded(x) #zx as noted in article
zy = model.get_encoded(y) #zy as noted in article

rm = riemannian_dgm(zx, zy, T, 2, 3, model.get_encoded, model.get_decoded, 100) #Defining class

loss, z_old, g_old, g_new, z_new = rm.geodesic_path_al1(alpha = 0.05, eps = 0.1) #Algorithm 1 in article

z_old = rm.get_list_to_torch(z_old, [0,1]) #Converting list to tensors in matrix-like format
z_new = rm.get_list_to_torch(z_new, [0,1]) #Converting list to tensors in matrix-like format

g_old = rm.get_list_to_torch(g_old, [0,1,2]) #Converting list to tensors in matrix-like format
g_new = rm.get_list_to_torch(g_new, [0,1,2]) #Converting list to tensors in matrix-like format

#%% Riemannian Plots

#Plotting loss
data_plot.plot_loss(loss)

#Plotting in Z-dimension
x_old = z_old
x_new = z_new
x_old = x_old.detach().numpy()
x_new = x_new.detach().numpy()
data_plot.plot_geodesic_in_Z_2d([x_old, 'old'], [x_new, 'new'])

#Plotting on geodesic on the true manifold
x_old = g_old
x_new = g_new
x_old = x_old.detach().numpy()
x_new = x_new.detach().numpy()
data_plot.plot_geodesic_in_X_3d([-1,1], [-1,1], [x_old, 'old'], [x_new, 'new'])


