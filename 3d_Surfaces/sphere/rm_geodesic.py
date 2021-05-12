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
import numpy as np
import argparse
pi = math.pi

#Own files
from rm_com import riemannian_data
from VAE_surface3d import VAE_3d

#%% Fun

def x3_sphere(x1, x2):
    
    N = len(x1)
    x3 = np.random.normal(0, 1, N)
    
    r = np.sqrt(x1**2+x2**2+x3**2)
    
    return x1/r, x2/r, x3/r

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='Data/sphere.csv', 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/simple_geodesic/', 
                        type=str)
    parser.add_argument('--name', default='simple_geodesic',
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--MAX_ITER', default=100000,
                        type=int)
    parser.add_argument('--eps', default=0.1,
                        type=int)
    parser.add_argument('--T', default=100,
                        type=int)
    parser.add_argument('--alpha', default=1,
                        type=float)
    parser.add_argument('--lr', default=0.0001,
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/main/sphere_epoch_100000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    #Loading data
    df = pd.read_csv(args.data_path, index_col=0)
    DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)
    
    #Loading model
    model = VAE_3d().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Latent coordinates
    zx = (torch.tensor([-3,-3])).float()
    zy = (torch.tensor([3,-3])).float()
    
    #Coordinates on the manifold
    x = (torch.tensor(x3_sphere(zx[0],zx[1]))).float()
    y = (torch.tensor(x3_sphere(zy[0],zy[1]))).float()
    
    #Mean of approximate posterier
    hx = model.h(x)[0]
    hy = model.h(y)[0]
    
    #Output of the mean
    gx = model.g(hx)
    gy = model.g(hy)
    
    #Loading module
    rm = riemannian_data(model.h, model.g, T=args.T, 
                         eps = args.eps, MAX_ITER=args.MAX_ITER)
    
    gamma_linear = rm.interpolate(hx, hy)
    gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = args.alpha,
                                          print_conv=True)
    
    checkpoint = args.save_path+args.name+'.pt'
    torch.save({'loss': gamma_geodesic[0],
                'E_fun': gamma_geodesic[1],
                'Z_old': gamma_linear,
                'Z_new': gamma_geodesic[2],
                'G_old': gamma_geodesic[3],
                'G_new': gamma_geodesic[4],
                'L_old': gamma_geodesic[5].item(),
                'L_new': gamma_geodesic[6].item(),
                'gx': gx,
                'gy': gy,
                'hx': hx,
                'hy': hy}, 
               checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    