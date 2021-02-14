# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 00:29:19 2021

@author: Frederik
"""

import torch
import torch.optim as optim

#Own files
from VAE_3d import VAE_3d_surface
from plot_dat import plot_3d_fun
from sim_dat import sim_3d_fun


#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

#%% Plotting

file_path_name = 'Data/para_data.csv'
file_model_save = 'trained_models/para_3d.pt'
data_obj = sim_3d_fun(N_sim=50000, name_path = file_path_name)
data_plot = plot_3d_fun(N_grid=100)
generate_data = False
load_trained_model = True
plot_data = True
try_cuda = False

epochs = 10#100000
batch_size = 100
lr = 0.0001

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

if generate_data:
    data_obj.sim_3d()

dat = data_obj.read_data()

if plot_data:
    x1 = dat[:,0].detach().numpy()
    x2 = dat[:,1].detach().numpy()
    x3 = dat[:,2].detach().numpy()
    #Plotting the true surface
    data_plot.true_plot_3d([min(x1), max(x1)], [min(x2), max(x2)])
    #Plotting the true surface with the simulated data
    data_plot.plot_data_scatter_3d(x1, x2, x3)
    #Surface plot of the data
    data_plot.plot_data_surface_3d(x1, x2, x3)

if load_trained_model:
    model = VAE_3d_surface(device = device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    checkpoint = torch.load(file_model_save)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    model.eval()
    
    X = model(dat)
    X_rec = X[1] #Index 1 is the reconstruction
    x1 = X_rec[:,0].detach().numpy()
    x2 = X_rec[:,1].detach().numpy()
    x3 = X_rec[:,2].detach().numpy()
    
    #Plotting loss function
    data_plot.plot_loss(loss)
    
    #Surface plot of the reconstructed data
    data_plot.plot_data_surface_3d(x1, x2, x3)
    
    #Plotting the true surface with the reconstructed data
    data_plot.plot_data_scatter_3d(x1, x2, x3, title='Scatter of Reconstructed Data')
    
    
    
    
    
    
    
    

