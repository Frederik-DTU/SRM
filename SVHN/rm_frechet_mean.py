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
import argparse

#Own files
from rm_computations import rm_data
from VAE_svhn import VAE_SVHN

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='Data_groups/group1.pt', 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/frechet_group1.pt', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=10000,
                        type=int)
    parser.add_argument('--T', default=10,
                        type=int)
    parser.add_argument('--batch_size', default=10,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    parser.add_argument('--size', default=32,
                        type=int)
    
    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/main/svhn_epoch_50000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    #Load
    DATA = torch.load(args.data_path)[0:args.batch_size]
    
    #Plotting the trained model
    model = VAE_SVHN().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    Z = model.h(DATA)
    
    muz_linear, mug_linear = rm.compute_euclidean_mean(Z)
    loss, muz_geodesic = rm.compute_frechet_mean_hpc(Z, muz_linear, args.save_path,
                                                     T = args.T,
                                                     epochs_geodesic = 100000,
                                                     epochs_frechet = 100000,
                                                     save_step = 10)
    mug_geodesic = model.g(muz_geodesic)
    
    torch.save({'loss': loss,
                 'muz_linear': muz_linear,
                 'mug_linear': mug_linear,
                 'muz_geodesic': muz_geodesic,
                 'mug_geodesic': mug_geodesic,
                 'T': args.T}, 
                args.save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    
