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
from torch.utils.data import DataLoader
import argparse

#Own files
from rm_com import riemannian_data
from VAE_celeba import VAE_CELEBA

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='Data_groups/group_blond_closed/', 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/frechet_mean/group_blond_closed.pt', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--MAX_ITER', default=10,
                        type=int)
    parser.add_argument('--eps', default=0.1,
                        type=int)
    parser.add_argument('--T', default=10,
                        type=int)
    parser.add_argument('--alpha', default=1,
                        type=float)
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
    
    trainloader = DataLoader(dataset, batch_size=100,
                         shuffle=False, num_workers=0)
    
    DATA = next(iter(trainloader))
    
    #Plotting the trained model
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Loading module
    rm = riemannian_data(model.h, model.g, T=args.T, 
                         eps = args.eps, MAX_ITER=args.MAX_ITER)
    
    Z = model.h(DATA[0])[0]
    
    L, muz_init, mug_init, mu_z, mu_g = rm.get_frechet_mean(Z, alpha_mu = 0.1, 
                                                            alpha_g = 0.1,
                                                            print_conv=True)
    
    torch.save({'L': L,
                 'muz_init': muz_init,
                 'mug_init': mug_init,
                 'mu_z': mu_z,
                 'mu_g': mu_g}, 
                args.save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    