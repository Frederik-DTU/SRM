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

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse
import numpy as np

#Own files
from rm_computations import rm_data
from VAE_celeba import VAE_CELEBA

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/triangle.pt', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100000,
                        type=int)
    parser.add_argument('--T', default=10,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    parser.add_argument('--size', default=64,
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/main/celeba_epoch_6300.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    #Load
    dataset = dset.ImageFolder(root=args.data_path,
                           transform=transforms.Compose([
                               transforms.Resize(args.size),
                               transforms.CenterCrop(args.size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    
    #Plotting the trained model
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Get 3 images to compute geodesics for
    subset_indices = [0, 1, 2] # select your indices here as a list
    dataset_subset = torch.utils.data.Subset(dataset, subset_indices)
    
    xa = dataset_subset[0][0].view(1,3,args.size,args.size)
    xb = dataset_subset[1][0].view(1,3,args.size,args.size)
    xc = dataset_subset[2][0].view(1,3,args.size,args.size)
    
    za = model.h(xa)
    zb = model.h(xb)
    zc = model.h(xc)
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    _, _, zab_geodesic, gab_geodesic = rm.Log_map(za, zb, T = args.T, 
                          epochs=args.epochs)
    _, _, zac_geodesic, gac_geodesic = rm.Log_map(za, zc, T = args.T, 
                          epochs=args.epochs)
    _, _, zbc_geodesic, gbc_geodesic = rm.Log_map(zb, zc, T = args.T, 
                          epochs=args.epochs)
    
    vab_z = args.T*(zab_geodesic[1]-zab_geodesic[0]).view(-1)
    vac_z = args.T*(zac_geodesic[1]-zac_geodesic[0]).view(-1)
    vbc_z = args.T*(zbc_geodesic[1]-zbc_geodesic[0]).view(-1)
    
    vcb_z = -args.T*(zbc_geodesic[-1]-zbc_geodesic[-2]).view(-1)
    vca_z = -args.T*(zac_geodesic[-1]-zac_geodesic[-2]).view(-1)
    vba_z = -args.T*(zab_geodesic[-1]-zab_geodesic[-2]).view(-1)
    
    
    a_angle = torch.dot(vac_z, vab_z)/(torch.norm(vac_z)*torch.norm(vab_z))
    b_angle = torch.dot(vbc_z, vba_z)/(torch.norm(vbc_z)*torch.norm(vba_z))
    c_angle = torch.dot(vca_z, vcb_z)/(torch.norm(vca_z)*torch.norm(vcb_z))
    
    torch.save({'gab_geodesic': gab_geodesic,
                'gac_geodesic': gac_geodesic,
                'gbc_geodesic': gbc_geodesic,
                'a_angle': a_angle,
                'b_angle': b_angle,
                'c_angle': c_angle}, 
               args.save_path)
    
    
    

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    