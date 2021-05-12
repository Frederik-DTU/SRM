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

#Modules
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

#%% Loading data and model

dataroot = "../../Data/SVHN"
file_model_save = 'trained_models/svhn_epoch_50000.pt'
save_groups = 'Data_groups/'
device = 'cpu'
lr = 0.0002
group_size = 50
num_groups = 10
names = ['group1.pt',
         'group2.pt',
         'group3.pt',
         'group4.pt',
         'group5.pt',
         'group6.pt',
         'group7.pt',
         'group8.pt',
         'group9.pt',
         'group10.pt']

dataset = dset.SVHN(root=dataroot, split = 'train',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]))

trainloader = DataLoader(dataset, batch_size=group_size,
                         shuffle=False, num_workers=0)

count = 0
for x in trainloader:
    torch.save(x[0], save_groups+names[count])
    count += 1
    
    if count>=num_groups:
        break
