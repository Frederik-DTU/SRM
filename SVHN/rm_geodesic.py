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

#Own files
from rm_com import riemannian_data
from VAE_svhn import VAE_SVHN

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default='../../Data/SVHN', 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/simple_geodesic/', 
                        type=str)
    parser.add_argument('--name', default='simple_geodesic',
                        type=str)

    #Hyper-parameters
    parser.add_argument('--device', default='cpu', #'cuda:0'
                        type=str)
    parser.add_argument('--MAX_ITER', default=100,
                        type=int)
    parser.add_argument('--eps', default=0.1,
                        type=int)
    parser.add_argument('--T', default=10,
                        type=int)
    parser.add_argument('--alpha', default=1,
                        type=float)
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
    
    #Load
    dataset = dset.SVHN(root=args.data_path, split = 'train',
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                               ]))
    
    #Plotting the trained model
    model = VAE_SVHN().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Get 3 images to compute geodesics for
    subset_indices = [0, 1, 2, 3, 4, 5] # select your indices here as a list
    dataset_subset = torch.utils.data.Subset(dataset, subset_indices)
    
    #Loading module
    rm = riemannian_data(model.h, model.g, T=args.T, 
                         eps = args.eps, MAX_ITER=args.MAX_ITER)
    
    img_height = args.size+2
    img_n = len(subset_indices)
    
    G_plot = torch.empty(1)
    
    for i in range(int(img_n/2)):
        x = (dataset_subset[2*i][0]).view(1, 3, args.size, args.size)
        y = (dataset_subset[2*i+1][0]).view(1, 3, args.size, args.size)
        
        hx = model.h(x)[0]
        hy = model.h(y)[0]
        
        gamma_linear = rm.interpolate(hx, hy)
        gamma_geodesic = rm.geodesic_path_al1(gamma_linear, alpha = 0.1,
                                              print_conv = True)
            
        G_plot = torch.cat((gamma_geodesic[3].detach(), gamma_geodesic[7].detach()), dim = 0)
        
        arc_length = ['{0:.4f}'.format(gamma_geodesic[5]), '{0:.4f}'.format(gamma_geodesic[6])]
        tick_list = [img_height/2, img_height/2+img_height]
        
        torch.save({'G_plot': G_plot,
                    'arc_length': arc_length,
                    'tick_list': tick_list,
                    'T': args.T}, 
                   args.save_path+'geodesic'+str(i+1)+'.pt')

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    