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

#%% Plotting

dataroot = "../../Data/SVHN" #Directory for dataset
file_model_save = 'trained_models/svhn_epoch_10.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
device = 'cpu'
lr = 0.0002

dataset = dset.SVHN(root=dataroot, split = 'extra',
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
plt.imshow(np.transpose(vutils.make_grid(x_hat[1].to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()