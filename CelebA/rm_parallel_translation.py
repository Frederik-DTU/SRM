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
    
    data_path = 'Data_groups/group_blond_closed/'
    img_size = 64
    
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=1,
                         shuffle=False, num_workers=0)
    blond_closed = next(iter(trainloader))[0]
    
    data_path = 'Data_groups/group_blond_open/'
    img_size = 64
    
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        
    trainloader = DataLoader(dataset, batch_size=2,
                         shuffle=False, num_workers=0)
    blond_closed_b = next(iter(trainloader))[0]
    blond_closed_c = blond_closed_b[1].view(1,3,64,64)
    blond_closed_b = blond_closed_b[0].view(1,3,64,64)
    
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    z_blond_closed = model.h(blond_closed)
    z_blond_closed_b = model.h(blond_closed_b)
    z_blond_closed_c = model.h(blond_closed_c)
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    v_z, v_g = rm.Log_map(z_blond_closed, z_blond_closed_b, T = args.T)
    v_z, v_g = v_z.view(1,-1), v_g.view(1,3,64,64)
    
    z_linear_a_c = rm.interpolate(z_blond_closed, z_blond_closed_c, args.T)
    _, z_geodesic_a_c = rm.compute_geodesic(z_linear_a_c)
    
    #https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
    vT_a_c, uT_a_c = rm.parallel_translation_al2(z_geodesic_a_c, v_z)
    
    Z_a_c, G_a_c = rm.geodesic_shooting_al3(z_blond_closed_c, uT_a_c, T = 10)
    
    
    plot_img = G_a_c[-1].view(3,64,64).detach()
    plt.figure(figsize=(8,6))
    plt.axis("off")
    plt.title("Linear mean")
    plt.imshow(plot_img.permute(1, 2, 0))
    
    
    plot_img2 = blond_closed.view(3,64,64).detach()
    plot_img3 = blond_closed_b.view(3,64,64).detach()
    plot_img4 = blond_closed_c.view(3,64,64).detach()
    
    plt.figure(figsize=(8,6))
    plt.axis("off")
    plt.title("Linear mean")
    plt.imshow(plot_img4.permute(1, 2, 0))
    
    plt.figure(figsize=(8,6))
    plt.axis("off")
    plt.title("Original Images")
    plt.imshow(vutils.make_grid(G_a_c.detach(), padding=2, normalize=True, nrow=11).permute(1, 2, 0))
    
    

    return

#%% Calling main

if __name__ == '__main__':
    main()

    
    