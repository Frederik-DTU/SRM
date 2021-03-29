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
from rm_com_v2 import riemannian_data
from VAE_celeba import VAE_CELEBA

#%% Loading data and model

dataroot = "../../Data/CelebA/celeba" #Directory for dataset
file_model_save = 'trained_models/celeba_epoch_600.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
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

#%% Getting points on the learned manifold

#Get 3 images to compute geodesics for
subset_indices = [0, 1, 2, 3, 4, 5] # select your indices here as a list
dataset_subset = torch.utils.data.Subset(dataset, subset_indices)

#%% Getting geodesic between learned points

#Parameters for the Riemannian computations
T = 10
eps = 0.1

#Loading module
rm = riemannian_data(model.h, model.g, T = T, eps = eps, MAX_ITER=100,
                     div_fac=2)

#%%Estimating geodesic between points

img_height = img_size+2,
img_n = len(subset_indices)

G_plot = []
lst_midpoint = []

i = 0 #remove until for loop start
    
x = (dataset_subset[2*i][0]).view(1, 3, img_size, img_size)
y = (dataset_subset[2*i+1][0]).view(1, 3, img_size, img_size)

hx = model.h(x)[0]
hy = model.h(y)[0]

gamma_linear = rm.interpolate(hx, hy)
gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = 0.1, print_conv = True)
u0 = gamma_geodesic[4]
u0 = (u0[1]-u0[0]/T).view(-1)
test = rm.parallel_translation_al2(gamma_linear, u0)

#Remove above

for i in range(int(img_n/2)):
    
    x = (dataset_subset[2*i][0]).view(1, 3, img_size, img_size)
    y = (dataset_subset[2*i+1][0]).view(1, 3, img_size, img_size)
    
    hx = model.h(x)[0]
    hy = model.h(y)[0]
    
    gamma_linear = rm.interpolate(hx, hy)
    gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = 0.1, print_conv = True)

    G_old = (data_plot.cat_tensors(gamma_geodesic[3])).detach()
    G_new = (data_plot.cat_tensors(gamma_geodesic[4])).detach()
        
    G_plot.append(torch.cat((G_old, G_new), dim = 0))
    
    x = (G_old[int(T/2)]).view(1, 3, img_size, img_size)
    y = (G_new[int(T/2)]).view(1, 3, img_size, img_size)
    lst_midpoint.append(torch.cat((x, y), dim = 0))

    arc_length = ['{0:.4f}'.format(gamma_geodesic[5]), '{0:.4f}'.format(gamma_geodesic[6])]
    tick_list = [img_height/2, img_height/2+img_height]
    
fig, ax = plt.subplots(3,1, figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")
for i in range(int(img_n/2)):
    
    ax[i].imshow(vutils.make_grid(G_plot[i], padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
    ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(tick_list)
    ax[i].set_yticklabels(arc_length)

N = len(lst_midpoint)
for i in range(N):
    plt.figure(figsize=(8,6))
    plt.imshow(np.transpose(vutils.make_grid(lst_midpoint[i], padding=2, normalize=True, nrow=T+1),
                            (1,2,0)))

#%% Computing Fréchet mean for sample

#https://fairyonice.github.io/Welcome-to-CelebA.html

#%% Geodesic Distance matrix

    
    
    
    