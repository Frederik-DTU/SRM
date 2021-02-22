# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 12:34:27 2021

@author: Frederik
"""
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
from VAE_test import vae_para_test


#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

#%% Data generator

def x3_fun(x1, x2):
    
    return 0*x1

#%% Plotting

test_model = False
file_path_name = 'Data/surface_R2.csv'
file_model_save = 'trained_models/surface_R2.pt'
data_obj = sim_3d_fun(N_sim=50000, name_path = file_path_name, x3_fun = x3_fun)
data_plot = plot_3d_fun(N_grid=100, fun=x3_fun)
generate_data = False
load_trained_model = True
plot_data = True
try_cuda = False

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
    if test_model:
        model = vae_para_test(device = device).to(device)
    else:
        model = VAE_3d_surface(device = device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    checkpoint = torch.load(file_model_save, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    bce_loss = checkpoint['BCE_loss']
    kld_loss = checkpoint['KLD_loss']
    
    model.eval()
    
    X = model(dat)
    z = X[0]
    mu = X[2]
    var = X[3]
    X_rec = X[1] #Index 1 is the reconstruction
    x1 = X_rec[:,0].detach().numpy()
    x2 = X_rec[:,1].detach().numpy()
    x3 = X_rec[:,2].detach().numpy()
    
    #Plotting loss function
    data_plot.plot_loss(loss, title='Loss function')
    data_plot.plot_loss(bce_loss, title='BCE Loss')
    data_plot.plot_loss(kld_loss, title='KLD Loss')
    
    #Surface plot of the reconstructed data
    data_plot.plot_data_surface_3d(x1, x2, x3)
    
    #Plotting the true surface with the reconstructed data
    data_plot.plot_data_scatter_3d(x1, x2, x3, title='Scatter of Reconstructed Data')
    
    #Plotting mu in Z-space
    z = z.detach().numpy()
    mu = mu.detach().numpy()
    var = var.detach().numpy()
    data_plot.plot_dat_in_Z_2d([z, 'z'])
    data_plot.plot_dat_in_Z_2d([mu, 'mu'])
    data_plot.plot_dat_in_Z_2d([var, 'var'])
    
    
    
    
    
    
    
    
    

