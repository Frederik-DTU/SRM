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

#%% Fun

def x3_hyper_para(x1, x2):
    
    return x1, x2, x1**2-x2**2

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='Data/hyper_para.csv', 
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
    parser.add_argument('--load_model_path', default='trained_models/main/hyper_para_epoch_100000.pt',
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
    x = (torch.tensor(x3_hyper_para(zx[0],zx[1]))).float()
    y = (torch.tensor(x3_hyper_para(zy[0],zy[1]))).float()
    
    #Mean of approximate posterier
    hx = model.h(x)
    hy = model.h(y)
    
    #Output of the mean
    gx = model.g(hx)
    gy = model.g(hy)
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    z_linear = rm.interpolate(hx, hy, args.T)
    loss, z_geodesic = rm.compute_geodesic(z_linear, 100000)
    g_linear = model.g(z_linear)
    g_geodesic = model.g(z_geodesic)
    L_linear = rm.arc_length(g_linear)
    L_geodesic = rm.arc_length(g_geodesic)
    
    checkpoint = args.save_path+args.name+'.pt'
    torch.save({'loss': loss,
                'z_linear': z_linear,
                'z_geodesic': z_geodesic,
                'g_linear': g_linear,
                'g_geodesic': g_geodesic,
                'L_linear': L_linear.item(),
                'L_geodesic': L_geodesic.item(),
                'gx': gx,
                'gy': gy,
                'hx': hx,
                'hy': hy}, 
               checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    