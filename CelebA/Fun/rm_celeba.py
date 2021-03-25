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
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

#Own files
from plot_dat import plot_3d_fun
from rm_com import riemannian_data
from VAE_celeba import VAE_CELEBA

#%% Loading data and model

dataroot = "Data" #Directory for dataset
file_model_save = 'trained_models/fun_epoch_10000.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
device = 'cpu'
lr = 0.0002

img_size = 64
data_plot = plot_3d_fun(N_grid=100) #x3_hyper_para

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

trainloader = DataLoader(dataset, batch_size=4,
                         shuffle=False, num_workers=0)

#Plotting the trained model
model = VAE_CELEBA().to(device) #Model used
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint = torch.load(file_model_save, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
elbo = checkpoint['ELBO']
rec_loss = checkpoint['rec_loss']
kld_loss = checkpoint['KLD']

model.eval()

#%% Getting points on the learned manifold

#Get 3 images to compute geodesics for
subset_indices = [1, 2, 0, 3, 4, 5] # select your indices here as a list
dataset_subset = torch.utils.data.Subset(dataset, subset_indices)

test1 = dataset_subset[0][0]
test2 = dataset_subset[1][0]

test1 = test1.view(1, 3, 64, 64)
test2 = test2.view(1, 3, 64, 64)


#Mean of approximate posterier
hx = model.h(test1)[0]
hy = model.h(test2)[0]

#Output of the mean
gx = model.g(hx)
gy = model.g(hy)

#%% Getting geodesic between learned points

#Parameters for the Riemannian computations
T = 100
n_h = 32
n_g = 3

#Loading module
rm = riemannian_data(T, n_h, n_g, model.h, model.g, eps = 500)

#%%Estimating geodesic between points

gamma_linear = rm.interpolate(hx, hy)
gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = 0.0001)

num_mean = 100

G_old = data_plot.cat_tensors(gamma_geodesic[3])
G_oldplot = G_old.detach()
G_new = data_plot.cat_tensors(gamma_geodesic[4])
G_newplot = G_new.detach()
G_plot = torch.cat((G_newplot, G_oldplot), dim = 0)

fig = plt.figure(figsize=(8,6))
rows = ['test1', 'test2']
axes = plt.subplot(nrows = 2, 111)
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1),
                        (1,2,0)))
axes.set_xticklabels([''] + rows)

mid_idx = int(T/2)
mid_pic = G_plot[T+1+mid_idx]
fig = plt.figure(figsize=(2,1))
plt.imshow(np.transpose(vutils.make_grid(mid_pic, padding=2, normalize=True, nrow=T+1),
                        (1,2,0)))



frechet_means = rm.get_frechet_mean(Z[0:num_mean])

z_old = data_plot.convert_list_to_np(gamma_linear)
z_new = data_plot.convert_list_to_np(gamma_geodesic[2])
G_old = data_plot.convert_list_to_np(gamma_geodesic[3])
G_new = data_plot.convert_list_to_np(gamma_geodesic[4])

true_geodesic_mat = data_plot.convert_list_to_np(true_geodesic[4])

data_plot.plot_geodesic_in_Z_2d([z_old, 'Interpolation'], [z_new, 'Geodesic'])
data_plot.plot_geodesic_in_X_3d([-5, 5], [-5, 5], [G_old, 'Interpolation'], 
                                [G_new, 'Approximated Geodesic'], 
                                [true_geodesic_mat, 'True Geodesic'])

#%% Plotting true data

#Plotting the raw data
x1 = DATA[:,0].detach().numpy()
x2 = DATA[:,1].detach().numpy()
x3 = DATA[:,2].detach().numpy()
data_plot.true_plot_3d([min(x1), max(x1)], [min(x2), max(x2)]) #Plotting the true surface
data_plot.plot_data_scatter_3d(x1, x2, x3) #Plotting the true surface with the simulated data
data_plot.plot_data_surface_3d(x1, x2, x3) #Surface plot of the data

#Plotting the trained model
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
data_plot.plot_data_scatter_3d(x1, x2, x3, title='Scatter of Reconstructed Data')

#Plotting mu in Z-space
z = z.detach().numpy()
mu = mu.detach().numpy()
std = std.detach().numpy()
data_plot.plot_dat_in_Z_2d([z, 'z'])
data_plot.plot_dat_in_Z_2d([mu, 'mu'])
data_plot.plot_dat_in_Z_2d([std, 'std'])
    
    
    