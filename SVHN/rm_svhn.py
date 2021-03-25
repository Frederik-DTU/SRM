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
import random

#Own files
from plot_dat import plot_3d_fun
from rm_com import riemannian_data
from VAE_svhn import VAE_SVHN

#%% Loading data and model

dataroot = "../../Data/SVHN"
file_model_save = 'trained_models/svhn_epoch_5000.pt'
device = 'cpu'
lr = 0.0002

data_plot = plot_3d_fun(N_grid=100)
dataset = dset.SVHN(root=dataroot, split = 'train',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))

trainloader = DataLoader(dataset, batch_size=64,
                         shuffle=False, num_workers=0)

#Plotting the trained model
model = VAE_SVHN().to(device) #Model used
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
subset_indices = [0, 1, 2, 3, 4, 5] # select your indices here as a list
dataset_subset = torch.utils.data.Subset(dataset, subset_indices)

#%% Getting geodesic between learned points

#Parameters for the Riemannian computations
T = 10

#Loading module
rm = riemannian_data(model.h, model.g, T = 10, eps = 0.1)

#%%Estimating geodesic between points

img_height = 34
img_n = len(subset_indices)

G_plot = torch.empty(1)

fig, ax = plt.subplots(3,1, figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")
for i in range(int(img_n/2)):
    x = (dataset_subset[2*i][0]).view(1, 3, 32, 32)
    y = (dataset_subset[2*i+1][0]).view(1, 3, 32, 32)
    
    hx = model.h(x)[0]
    hy = model.h(y)[0]
    
    gamma_linear = rm.interpolate(hx, hy)
    gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = 0.1, print_conv = True)

    G_old = (data_plot.cat_tensors(gamma_geodesic[3])).detach()
    G_new = (data_plot.cat_tensors(gamma_geodesic[4])).detach()
        
    G_plot = torch.cat((G_old, G_new), dim = 0)

    arc_length = ['{0:.4f}'.format(gamma_geodesic[5]), '{0:.4f}'.format(gamma_geodesic[6])]
    tick_list = [img_height/2, img_height/2+img_height]
    
    ax[i].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(tick_list)
    ax[i].set_yticklabels(arc_length)

#%% Computing Fréchet mean for sample

random.seed(100)

rm = riemannian_data(model.h, model.g, T = 10, eps = 1)
mean_sample = 64
N = len(dataset)

rnd_idx = np.random.choice(N, mean_sample, replace=False)
dataset_subset = torch.utils.data.Subset(dataset, rnd_idx)
trainloader = DataLoader(dataset = dataset_subset, batch_size= mean_sample)

img = next(iter(trainloader))[0]
Z, _, _ = model.h(img)

frechet_means = rm.get_frechet_mean(Z, alpha_mu = 0.01)

linear_mean = (frechet_means[2].view(3,32,32)).detach()
geodesic_mean = (frechet_means[4].view(3,32,32)).detach()

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Linear mean of sample")
plt.imshow(linear_mean.permute(1,2,0))

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fréchet mean of sample")
plt.imshow(geodesic_mean.permute(1,2,0))
plt.show()

# Plot some training images
recon_batch = model(img) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
x_hat = recon_batch[1].detach()

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(img.to(device), padding=2, normalize=True).cpu(),(1,2,0)))

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Reconstruction Images")
plt.imshow(np.transpose(vutils.make_grid(x_hat.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

#%% Geodesic Distance matrix

    
    
    