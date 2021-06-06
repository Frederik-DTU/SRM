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
from VAE_celeba import VAE_CELEBA

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/dmat.pt', 
                        type=str)
    parser.add_argument('--group1', default='Data_groups/group_blond_closed/', 
                        type=str)
    parser.add_argument('--group2', default='Data_groups/group_blond_open/', 
                        type=str)
    parser.add_argument('--group3', default='Data_groups/group_black_closed/', 
                        type=str)
    parser.add_argument('--group4', default='Data_groups/group_black_open/', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100000,
                        type=int)
    parser.add_argument('--T', default=10,
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    parser.add_argument('--size', default=64,
                        type=int)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/main/celeba_epoch_6300.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    dataset = dset.ImageFolder(root=args.group1,
                               transform=transforms.Compose([
                                   transforms.Resize(args.size),
                                   transforms.CenterCrop(args.size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    
    x1 = next(iter(trainloader))[0].to(args.device)
    
    dataset = dset.ImageFolder(root=args.group2,
                               transform=transforms.Compose([
                                   transforms.Resize(args.size),
                                   transforms.CenterCrop(args.size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    
    x2 = next(iter(trainloader))[0].to(args.device)
    
    dataset = dset.ImageFolder(root=args.group3,
                               transform=transforms.Compose([
                                   transforms.Resize(args.size),
                                   transforms.CenterCrop(args.size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    
    x3 = next(iter(trainloader))[0].to(args.device)
    
    dataset = dset.ImageFolder(root=args.group4,
                               transform=transforms.Compose([
                                   transforms.Resize(args.size),
                                   transforms.CenterCrop(args.size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    
    x4 = next(iter(trainloader))[0].to(args.device)
    
    
    X = torch.cat((x1,x2,x3,x4), 0).to(args.device)
    X_list = [args.group1, args.group2, args.group3, args.group4]
    
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    Z = model.h(X)
    
    rm = rm_data(model.h, model.g, args.device)
    dmat = rm.geodesic_distance_matrix(Z, epochs=args.epochs, T=args.T)
    
    torch.save({'x_batch': X,
                'z_batch': Z,
                'dmat': dmat,
                'X_names': X_list}, 
               args.save_path)
    
    

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    
