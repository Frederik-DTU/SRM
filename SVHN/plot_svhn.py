# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 00:07:44 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
http://seaborn.pydata.org/generated/seaborn.jointplot.html
https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
https://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds
https://www.geeksforgeeks.org/scatter-plot-with-marginal-histograms-in-python-with-seaborn/
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
from sklearn.manifold import MDS
import seaborn as sns

#Own files
from VAE_svhn import VAE_SVHN
from plot_dat import plot_3d_fun
from rm_computations import rm_data

#%% Loading data and model

dataroot = "../../Data/SVHN/" #Directory for dataset
file_model_save = 'trained_models/main/svhn_epoch_50000.pt' #'trained_models/hyper_para/para_3d_epoch_100000.pt'
device = 'cpu'
lr = 0.0002

img_size = 32
data_plot = plot_3d_fun(N=100) #x3_hyper_para

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

load_path = 'rm_computations/'
names = ['simple_geodesic1.pt', 'simple_geodesic2.pt', 'simple_geodesic3.pt']
fig, ax = plt.subplots(3,1, figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")
img_height = img_size+2
for i in range(len(names)):
    
    tick_list_x = []
    euc_length_x = []
    
    checkpoint = torch.load(load_path+names[i])
    
    G_plot = checkpoint['G_plot']
    arc_length = checkpoint['arc_length']
    tick_list = checkpoint['tick_list']
    T = checkpoint['T']
    
    for j in range(T+1):
        tick_list_x.append(img_height/2+j*img_height)
        euc_length_x.append('{0:.3f}'.format(torch.norm((G_plot[j]-G_plot[j+T+1]).view(-1)).item()))
    
    G_plot = checkpoint['G_plot']
    ax[i].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
    #ax[i].axes.get_xaxis().set_visible(False)
    ax[i].set_yticks(tick_list)
    ax[i].set_yticklabels(arc_length)
    ax[i].set_xticks(tick_list_x)
    ax[i].set_xticklabels(euc_length_x) 

#%% Plotting Frechet mean for group

data_path = ['Data_groups/group1.pt', 'Data_groups/group2.pt',
             'Data_groups/group3.pt', 'Data_groups/group4.pt']
frechet_path = ['rm_computations/frechet_group1.pt', 'rm_computations/frechet_group2.pt',
                'rm_computations/frechet_group3.pt', 'rm_computations/frechet_group4.pt']
names = ['Group 1', 'Group 2',
         'Group 3', 'Group 4']
img_size = 32
batch_size = 10
img_height = img_size+2

tick_list_y = []
for j in range(len(data_path)):
    tick_list_y.append(img_height/2+j*img_height)

DATA = torch.empty(1)
for i in range(len(data_path)):
    path = data_path[i]
    
    if i == 0:
        DATA = torch.load(path)[0:batch_size]
    else:
        DATA = torch.cat((DATA,torch.load(path)[0:batch_size]), dim = 0)
    
fig, ax = plt.subplots(1,2,figsize=(8,6))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_title("Original Images")
ax[0].imshow(np.transpose(vutils.make_grid(DATA.to(device), padding=2, normalize=True, nrow=batch_size).cpu(),(1,2,0)))
ax[0].set_yticks(tick_list_y)
ax[0].set_yticklabels(names)

rec = model(DATA)[1].detach()

# Plot some training images
ax[1].axes.get_xaxis().set_visible(False)
ax[1].set_title("Reconstructed Images")
ax[1].imshow(np.transpose(vutils.make_grid(rec.to(device), padding=2, normalize=True, nrow=batch_size).cpu(),(1,2,0)))
ax[1].set_yticks(tick_list_y)
ax[1].set_yticklabels(names)

fig, ax = plt.subplots(1,1,figsize=(8,6))
ax.axes.get_xaxis().set_visible(False)
ax.set_title("Reconstructed Images")
ax.imshow(np.transpose(vutils.make_grid(rec.to(device), padding=2, normalize=True, nrow=batch_size).cpu(),(1,2,0)))
ax.set_yticks(tick_list_y)
ax.set_yticklabels(names)

fig, ax = plt.subplots(4,1,figsize=(8,6))
ax[0].set_title("Fréchet Mean")
tick_list_x = [img_height/2,img_height/2+img_height]
for i in range(len(frechet_path)):
    frechet = torch.load(frechet_path[i])
    mug_linear = frechet['mug_linear'].view(1,3,img_size,img_size).detach()
    mug_geodesic = frechet['mug_geodesic'].view(1,3,img_size,img_size).detach()
    loss = frechet['loss']
    G_plot = torch.cat((mug_linear.detach(), mug_geodesic.detach()), dim = 0)
    ax[i].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=2).permute(1, 2, 0))
    l_sum = ['{0:.3f}'.format(loss[0]),'{0:.3f}'.format(loss[-1])]
    ax[i].axes.get_yaxis().set_visible(False)
    ax[i].set_xticks(tick_list_x)
    ax[i].set_xticklabels(l_sum) 

#%% Plotting Parallel Transport

rm = rm_data(model.h, model.g, 'cpu')

#load_path = 'rm_computations/parallel_translation_1_2.pt'
load_path = 'rm_computations/parallel_translation_3_4.pt'

img_size = 32
batch_size = 10
img_height = img_size+2

checkpoint = torch.load(load_path)
T = checkpoint['T']
gab_geodesic = checkpoint['gab_geodesic']
gac_geodesic = checkpoint['g_ac']
gc_geodesic = checkpoint['gc_geodesic']
zc_linear = checkpoint['zc_linear']
gc_linear = checkpoint['gc_linear']

L_ab = rm.arc_length(gab_geodesic)
L_ac = rm.arc_length(gac_geodesic)
L_c = rm.arc_length(gc_geodesic)
L_linear_c = rm.arc_length(gc_linear)
arc_length = ['a-b: {0:.4f}'.format(L_ab), 'a-c: {0:.4f}'.format(L_ac)]
tick_list = [img_height/2, img_height/2+img_height]

G_plot = torch.cat((gab_geodesic.detach(), gac_geodesic.detach()), dim = 0)

fig, ax = plt.subplots(2,1,figsize=(8,6))
ax[0].set_title("Geodesic cuves and Linear interpolation between images")    

ax[0].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
ax[0].axes.get_xaxis().set_visible(False)
ax[0].set_yticks(tick_list)
ax[0].set_yticklabels(arc_length) 

G_plot = torch.cat((gc_linear.detach(), gc_geodesic.detach()), dim = 0)
arc_length = ['c_l: {0:.4f}'.format(L_linear_c), 'c_g: {0:.4f}'.format(L_c)]
tick_list_x = []
euc_length_x = []
for j in range(T+1):
        tick_list_x.append(img_height/2+j*img_height)
        euc_length_x.append('{0:.3f}'.format(torch.norm((G_plot[j]-G_plot[j+T+1]).view(-1)).item()))

ax[1].imshow(vutils.make_grid(G_plot, padding=2, normalize=True, nrow=T+1).permute(1, 2, 0))
#ax[1].axes.get_xaxis().set_visible(False)
ax[1].set_yticks(tick_list)
ax[1].set_yticklabels(arc_length) 
ax[1].set_xticks(tick_list_x)
ax[1].set_xticklabels(euc_length_x) 

#%% distance matrix
rm = rm_data(model.h, model.g, 'cpu')

load_path = 'rm_computations/dmat.pt'
checkpoint = torch.load(load_path)
x_batch = (checkpoint['x_batch'].view(checkpoint['x_batch'].shape[0],-1)).detach().numpy()
z_batch = checkpoint['z_batch']
dmat = checkpoint['dmat'].detach().numpy()
X_names = checkpoint['X_names']
dmat_linear = rm.linear_distance_matrix(z_batch, 10).detach().numpy()
z_batch = z_batch.detach().numpy()

X_names = ['Group1', 'Group2', 
           'Group3', 'Group4']

embedding = MDS(n_components=2, dissimilarity = 'precomputed')
x2d_linear = embedding.fit_transform(dmat_linear)
x2d_geodesic = embedding.fit_transform(dmat)
embedding = MDS(n_components=2)
x2d_euclidean = embedding.fit_transform(x_batch)

data_size = int(x2d_euclidean.shape[0]/len(X_names))

column_names = ['group', 'x1', 'x2']
df = pd.DataFrame(index=range(x2d_euclidean.shape[0]), columns=column_names)

group = [item for item in X_names for i in range(data_size)]
df['group'] = group
df['x1'] = x2d_geodesic[:,0]
df['x2'] = x2d_geodesic[:,1]
  
p = sns.jointplot(data=df, x="x1", y="x2", hue="group")
p.fig.suptitle("Geodesic Distances")
p.fig.tight_layout()
p.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

df['x1'] = x2d_linear[:,0]
df['x2'] = x2d_linear[:,1]
  
p = sns.jointplot(data=df, x="x1", y="x2", hue="group")
p.fig.suptitle("Linear Distances")
p.fig.tight_layout()
p.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

df['x1'] = x2d_euclidean[:,0]
df['x2'] = x2d_euclidean[:,1]
  
p = sns.jointplot(data=df, x="x1", y="x2", hue="group")
p.fig.suptitle("Euclidean Distances")
p.fig.tight_layout()
p.fig.subplots_adjust(top=0.95) # Reduce plot to make room 