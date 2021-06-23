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
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import torch
import torch.optim as optim
import pandas as pd
import math
import argparse
pi = math.pi

#Own files
from rm_computations import rm_data
from VAE_surface3d import VAE_3d

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_name', default='paraboloid', 
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--epochs', default=100,
                        type=int)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0001,
                        type=float)
    parser.add_argument('--save_step', default=1,
                        type=int)

    #Continue training or not
    parser.add_argument('--load_epoch', default='100000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    #Loading data
    data_path = 'Data/'+args.data_name+'.csv'
    load_path = 'trained_models/'+args.data_name+'/'+args.data_name+'_epoch_'+args.load_epoch
    save_path = 'rm_computations/'+args.data_name+'/'+'frechet_mean.pt'
    
    df = pd.read_csv(data_path, index_col=0)
    DATA = torch.Tensor(df.values).to(args.device)
    DATA = torch.transpose(DATA, 0, 1)
    
    #Loading model
    model = VAE_3d().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(load_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    Z = model.h(DATA)
    Z = Z[0:args.batch_size]
    
    muz_linear, mug_linear = rm.compute_euclidean_mean(Z)
    
    #loss, muz_geodesic = rm.compute_frechet_mean(Z, muz_linear, epochs_geodesic=100000,
    #                                             epochs_frechet=args.epochs,
    #                                             print_com = True,
    #                                             frechet_lr = 1e-3)
    
    #loss, muz_geodesic = rm.compute_frechet_mean(Z, muz_linear, epochs_geodesic=100000,
    #                                             epochs_frechet=args.epochs,
    #                                             print_com = True,
    #                                             frechet_lr = 1e-3)
    loss, muz_geodesic = rm.compute_frechet_mean_hpc(Z, muz_linear, save_path, 
                                                     T = args.T, epochs_geodesic = 100000, 
                                                     epochs_frechet = 100000, 
                                                     save_step = 10)
    
    mug_geodesic = model.g(muz_geodesic)
    
    torch.save({'loss': loss,
                 'muz_linear': muz_linear,
                 'mug_linear': mug_linear,
                 'muz_geodesic': muz_geodesic,
                 'mug_geodesic': mug_geodesic,
                 'batch_size': args.batch_size}, 
                save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    
