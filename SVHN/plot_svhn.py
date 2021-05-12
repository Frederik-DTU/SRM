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

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

#Own files
from VAE_svhn import VAE_SVHN
from plot_dat import plot_3d_fun

#%% Loading data and model

dataroot = "../../Data/SVHN"
file_model_save = 'trained_models/main/svhn_epoch_50000.pt'
device = 'cpu'
lr = 0.0002

data_plot = plot_3d_fun(N=100)
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

data_path = 'Data_groups/'
group_name = 'group1.pt'
frechet_path = 'rm_computations/frechet_mean/'

DATA = torch.load(data_path+group_name)

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original Images")
plt.imshow(np.transpose(vutils.make_grid(DATA.to(device), padding=2, normalize=True).cpu(),(1,2,0)))

rec = model(DATA)[1].detach()

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Reconstruction Images")
plt.imshow(np.transpose(vutils.make_grid(rec.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

frechet = torch.load(frechet_path+group_name)
mug_init = frechet['mug_init'].view(3,32,32).detach()
mu_g = frechet['mu_g'].view(3,32,32).detach()

plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Linear mean")
plt.imshow(mug_init.permute(1, 2, 0))

# Plot some training images
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Frechet mean")
plt.imshow(mu_g.permute(1, 2, 0))
plt.show()



