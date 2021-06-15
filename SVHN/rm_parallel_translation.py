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
import argparse

#Own files
from rm_computations import rm_data
from VAE_svhn import VAE_SVHN

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/parallel_translation_1_2.pt', 
                        type=str)
    parser.add_argument('--group1', default='Data_groups/group1.pt', 
                        type=str)
    parser.add_argument('--group2', default='Data_groups/group2.pt', 
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
    parser.add_argument('--size', default=32,
                        type=float)

    #Continue training or not
    parser.add_argument('--load_model_path', default='trained_models/main/svhn_epoch_50000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():
    
    #Arguments
    args = parse_args()
    
    xa = torch.load(args.group1)[0].view(1,3,args.size,args.size)
    xb = torch.load(args.group2)
    xc = xb[1].view(1,3,args.size,args.size)
    xb = xb[0].view(1,3,args.size,args.size)
    
    model = VAE_SVHN().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    z_a = model.h(xa)
    z_b = model.h(xb)
    z_c = model.h(xc)
        
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    zc_linear, gc_linear = rm.linear_parallel_translation(z_a, z_b, z_c, T=args.T)
    va_z, va_g, zab_geodesic, gab_geodesic = rm.Log_map(z_a, z_b, T = args.T, 
                          epochs=args.epochs)
    va_z, va_g = va_z.view(1,-1), va_g.view(1,3,args.size,args.size)
    
    z_init = rm.interpolate(z_a, z_c, T = args.T)
    _, z_ac = rm.compute_geodesic(z_init, epochs=args.epochs)
    g_ac = model.g(z_ac)
    
    vac_z, vac_g = rm.parallel_translation_al2(z_ac, va_z)
    zc_geodesic, gc_geodesic, uT = rm.geodesic_shooting_al3(z_c, vac_g, T = args.T)
    
    torch.save({'T': args.T,
                'va_z': va_z,
                'va_g': va_g,
                'zab_geodesic': zab_geodesic,
                'gab_geodesic': gab_geodesic,
                'z_ac': z_ac,
                'g_ac': g_ac,
                'vac_z': vac_z,
                'vac_g': vac_g,
                'zc_geodesic': zc_geodesic,
                'gc_geodesic': gc_geodesic,
                'zc_linear': zc_linear,
                'gc_linear': gc_linear}, 
               args.save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    