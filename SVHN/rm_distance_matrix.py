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
#https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
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
from torch.utils.data import DataLoader

#Own files
from rm_computations import rm_data
from VAE_svhn import VAE_SVHN

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/dmat.pt', 
                        type=str)
    parser.add_argument('--group1', default='Data_groups/group1.pt', 
                        type=str)
    parser.add_argument('--group2', default='Data_groups/group2.pt', 
                        type=str)
    parser.add_argument('--group3', default='Data_groups/group3.pt', 
                        type=str)
    parser.add_argument('--group4', default='Data_groups/group4.pt', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=1000,
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
    
    x1 = torch.load(args.group1)[0:args.batch_size]
    x2 = torch.load(args.group2)[0:args.batch_size]
    x3 = torch.load(args.group3)[0:args.batch_size]
    x4 = torch.load(args.group4)[0:args.batch_size]
    
    X = torch.cat((x1,x2,x3,x4), 0).to(args.device)
    X_list = [args.group1, args.group2, args.group3, args.group4]
    
    model = VAE_SVHN().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    Z = model.h(X)
    
    rm = rm_data(model.h, model.g, args.device)
    
    N = Z.shape[0]
    dmat = torch.zeros(N, N)
    dmat = rm.geodesic_distance_matrix_hpc(Z, args.save_path, dmat, 0,
                                           epochs=args.epochs, T=args.T)
    
    torch.save({'x_batch': X,
                'z_batch': Z,
                'dmat': dmat,
                'X_names': X_list}, 
               args.save_path)
    
    

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    
