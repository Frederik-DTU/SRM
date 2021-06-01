# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 00:07:44 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

#%% Modules

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import pandas as pd

#Own files
from VAE_celeba import VAE_CELEBA
from plot_dat import plot_3d_fun

#%% Loading data and model

dataroot = "../../Data/CelebA/celeba" #Directory for dataset
file_model_save = 'trained_models/main/celeba_epoch_6300.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
device = 'cpu'
lr = 0.0002

img_size = 64
data_plot = plot_3d_fun(N=100) #x3_hyper_para

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

trainloader = DataLoader(dataset, batch_size=64,
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

#%% Plotting

# Plot some training images
real_batch = next(iter(trainloader))
recon_batch = model(real_batch[0]) #x=z, x_hat, mu, var, kld.mean(), rec_loss.mean(), elbo
x_hat = recon_batch[1].detach()

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(),(1,2,0)))

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Reconstruction Images")
plt.imshow(np.transpose(vutils.make_grid(x_hat.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

#Plotting loss function
data_plot.plot_loss(elbo, title='Loss function')
data_plot.plot_loss(rec_loss, title='Reconstruction Loss')
data_plot.plot_loss(kld_loss, title='KLD')

#%% Plotting simple geodesics

load_path = 'rm_computations/simple_geodesic/'
names = ['geodesic1.pt', 'geodesic2.pt', 'geodesic3.pt']
fig, ax = plt.subplots(3,1, figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")
for i in range(len(names)):
    
    checkpoint = torch.load(load_path+names[i])
    
    G_plot = checkpoint['G_plot']
    arc_length = checkpoint['arc_length']
    tick_list = checkpoint['tick_list']
    T = checkpoint['T']
        
    ax[i].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(tick_list)
    ax[i].set_yticklabels(arc_length)
    
#%% Plotting Frechet mean for group

data_path = 'Data_groups/group_blond_closed/'
frechet_path = 'rm_computations/frechet_mean/group_blond_closed.pt'
img_size = 64

dataset = dset.ImageFolder(root=data_path,
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
trainloader = DataLoader(dataset, batch_size=100,
                     shuffle=False, num_workers=0)

DATA = next(iter(trainloader))[0]

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(DATA.to(device), padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))


rec = model(DATA)[1].detach()

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Reconstruction Images")
plt.imshow(np.transpose(vutils.make_grid(rec.to(device), padding=2, normalize=True, nrow=10).cpu(),(1,2,0)))
plt.show()

frechet = torch.load(frechet_path)
mug_linear = frechet['mug_linear'].view(3,img_size,img_size).detach()
mug_geodesic = frechet['mug_geodesic'].view(3,img_size,img_size).detach()

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Linear mean")
plt.imshow(mug_linear.permute(1, 2, 0))

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Frechet mean")
plt.imshow(mug_geodesic.permute(1, 2, 0))
plt.show()

