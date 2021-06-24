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
import numpy as np
pi = math.pi

#Own files
from rm_computations import rm_data
from VAE_surface3d import VAE_3d

#%% Fun


#def fun(x1, x2):
#    
#    theta = np.pi/4
#    
#    return x1, x2*np.cos(theta), x2*np.sin(theta)

def fun(x1, x2):
    
    return x1, x2, x1**2+x2**2

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_name', default='paraboloid', 
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
    parser.add_argument('--lr', default=0.0001,
                        type=float)

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
    save_path = 'rm_computations/'+args.data_name+'/'+'parallel_transport.pt'
    
    df = pd.read_csv(data_path, index_col=0)
    DATA = torch.Tensor(df.values)
    DATA = torch.transpose(DATA, 0, 1)
    
    #Loading model
    model = VAE_3d().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(load_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Latent coordinates
    za = (torch.tensor([-3.0,-3.0])).float()
    zb = (torch.tensor([3.0,-3.0])).float()
    zc = (torch.tensor([-3.0,3.0])).float()
    
    #Coordinates on the manifold
    a = (torch.tensor(fun(za[0],za[1]))).float()
    b = (torch.tensor(fun(zb[0],zb[1]))).float()
    c = (torch.tensor(fun(zc[0],zc[1]))).float()
    
    #Mean of approximate posterier
    ha = model.h(a)
    hb = model.h(b)
    hc = model.h(c)
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    zc_linear, gc_linear = rm.linear_parallel_translation(ha, hb, hc, T=args.T)
    va_z, va_g, zab_geodesic, gab_geodesic = rm.Log_map(ha, hb, epochs=args.epochs, 
                                                  T = args.T)
    
    z_init = rm.interpolate(ha, hc, T = args.T)
    _, z_ac = rm.compute_geodesic(z_init, epochs=args.epochs)
    g_ac = model.g(z_ac)
    
    vac_z, vac_g = rm.parallel_translation_al2(z_ac, va_z)
    
    zc_geodesic, gc_geodesic, uT = rm.geodesic_shooting_al3(hc, vac_g, T = args.T)
    
    torch.save({'va_z': va_z,
                'va_g': va_g,
                'zab_geodesic': zab_geodesic,
                'gab_geodesic': gab_geodesic,
                'zac_geodesic': z_ac,
                'gac_geodesic': g_ac,
                'vac_z': vac_z,
                'vac_g': vac_g,
                'zc_geodesic': zc_geodesic,
                'gc_geodesic': gc_geodesic,
                'zc_linear': zc_linear,
                'gc_linear': gc_linear,
                'a': a,
                'b': b,
                'c': c,
                'uT': uT}, 
               save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    
    