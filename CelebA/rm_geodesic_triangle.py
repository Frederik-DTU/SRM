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
import numpy as np

#Own files
from rm_computations import rm_data
from VAE_celeba import VAE_CELEBA

#%% Parser for command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--data_path', default="../../Data/CelebA/celeba", 
                        type=str)
    parser.add_argument('--save_path', default='rm_computations/triangle', 
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
    
    #Plotting the trained model
    model = VAE_CELEBA().to(args.device) #Model used
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load(args.load_model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.eval()
    
    #Get 3 images to compute geodesics for
    subset_indices = [0, 1, 2] # select your indices here as a list
    dataset_subset = torch.utils.data.Subset(dataset, subset_indices)
    
    xa = dataset_subset[0][0].view(1,3,args.size,args.size)
    xb = dataset_subset[1][0].view(1,3,args.size,args.size)
    xc = dataset_subset[2][0].view(1,3,args.size,args.size)
    
    za = model.h(xa)
    zb = model.h(xb)
    zc = model.h(xc)
    
    #Loading module
    rm = rm_data(model.h, model.g, args.device)
    
    gamma_linear = rm.interpolate(za, zb, args.T)
    _, zab_geodesic = rm.compute_geodesic(gamma_linear,epochs=args.epochs)
    gab_geodesic = model.g(zab_geodesic)
    gab_linear = model.g(gamma_linear)
    
    gamma_linear = rm.interpolate(za, zc, args.T)
    _, zac_geodesic = rm.compute_geodesic(gamma_linear,epochs=args.epochs)
    gac_geodesic = model.g(zac_geodesic)
    gac_linear = model.g(gamma_linear)
    
    gamma_linear = rm.interpolate(zb, zc, args.T)
    _, zbc_geodesic = rm.compute_geodesic(gamma_linear,epochs=args.epochs)
    gbc_geodesic = model.g(zbc_geodesic)
    gbc_linear = model.g(gamma_linear)
    
    vab_z = args.T*(zab_geodesic[1]-zab_geodesic[0]).view(-1)
    vac_z = args.T*(zac_geodesic[1]-zac_geodesic[0]).view(-1)
    vbc_z = args.T*(zbc_geodesic[1]-zbc_geodesic[0]).view(-1)
    
    vcb_z = -args.T*(zbc_geodesic[-1]-zbc_geodesic[-2]).view(-1)
    vca_z = -args.T*(zac_geodesic[-1]-zac_geodesic[-2]).view(-1)
    vba_z = -args.T*(zab_geodesic[-1]-zab_geodesic[-2]).view(-1)
    
    
    a_angle = np.arccos((torch.dot(vac_z, vab_z)/(torch.norm(vac_z)*torch.norm(vab_z))).detach().numpy())
    b_angle = np.arccos((torch.dot(vbc_z, vba_z)/(torch.norm(vbc_z)*torch.norm(vba_z))).detach().numpy())
    c_angle = np.arccos((torch.dot(vca_z, vcb_z)/(torch.norm(vca_z)*torch.norm(vcb_z))).detach().numpy())
    
    L_ab = rm.arc_length(gab_geodesic)
    L_ac = rm.arc_length(gac_geodesic)
    L_bc = rm.arc_length(gbc_geodesic)
    
    L_ab_linear = rm.arc_length(gab_linear)
    L_ac_linear = rm.arc_length(gac_linear)
    L_bc_linear = rm.arc_length(gbc_linear)
            
    save_path = args.save_path+'_simple.pt'
    torch.save({'gab_geodesic': gab_geodesic,
                'gac_geodesic': gac_geodesic,
                'gbc_geodesic': gbc_geodesic,
                'L_ab': L_ab.item(),
                'L_ac': L_ac.item(),
                'L_bc': L_bc.item(),
                'gab_linear': gab_linear,
                'gac_linear': gac_linear,
                'gbc_linear': gbc_linear,
                'L_ab_linear': L_ab_linear.item(),
                'L_ac_linear': L_ac_linear.item(),
                'L_bc_linear': L_bc_linear.item(),
                'a_angle': a_angle,
                'b_angle': b_angle,
                'c_angle': c_angle}, 
               save_path)
    
    load_path = 'rm_computations/dmat.pt'
    checkpoint = torch.load(load_path)
    x_batch = checkpoint['x_batch']
    dmat = checkpoint['dmat'].detach().numpy()
    
    tol = 2.5
    
    N = dmat.shape[0]
    for j in range(2,N):
        idx_list = [j]
        row_dmat = dmat[j]
        for i in range(1,N):
            val = row_dmat[i]
            row_dmat2 = dmat[i]
            dmat_restricted = row_dmat2[(row_dmat2<val+tol) & (row_dmat2>val-tol)]
            if len(dmat_restricted)>0:
                val_list = [val]
                for iter_val in dmat_restricted:
                    if len(idx_list) == 3:
                        break
                    if not (iter_val in val_list):
                        idx = np.where(iter_val == row_dmat2)[0][0]
                        if (row_dmat[idx]<val+tol) and (row_dmat[idx]>val-tol) and (row_dmat[idx]>0.0):
                            idx_list.append(i)
                            val_list.append(iter_val)
                            idx_list.append(idx)
            if len(idx_list) == 3:
                    break
        if len(idx_list) == 3:
                    break
        else:
            idx_list = []
            val_list = []
                
    
    xa = x_batch[idx_list[0]].view(1,3,args.size,args.size)
    xb = x_batch[idx_list[1]].view(1,3,args.size,args.size)
    xc = x_batch[idx_list[2]].view(1,3,args.size,args.size)
    
    za = model.h(xa)
    zb = model.h(xb)
    zc = model.h(xc)
    
    #Loading module
    zab_linear = rm.interpolate(za, zb, args.T)
    _, zab_geodesic = rm.compute_geodesic(zab_linear,epochs=args.epochs)
    gab_geodesic = model.g(zab_geodesic)
    gab_linear = model.g(zab_linear)
    
    zac_linear = rm.interpolate(za, zc, args.T)
    _, zac_geodesic = rm.compute_geodesic(zac_linear,epochs=args.epochs)
    gac_geodesic = model.g(zac_geodesic)
    gac_linear = model.g(zac_linear)
    
    zbc_linear = rm.interpolate(zb, zc, args.T)
    _, zbc_geodesic = rm.compute_geodesic(zbc_linear,epochs=args.epochs)
    gbc_geodesic = model.g(zbc_geodesic)
    gbc_linear = model.g(zbc_linear)
    
    vab_z = args.T*(zab_geodesic[1]-zab_geodesic[0]).view(-1)
    vac_z = args.T*(zac_geodesic[1]-zac_geodesic[0]).view(-1)
    vbc_z = args.T*(zbc_geodesic[1]-zbc_geodesic[0]).view(-1)
    
    vcb_z = -args.T*(zbc_geodesic[-1]-zbc_geodesic[-2]).view(-1)
    vca_z = -args.T*(zac_geodesic[-1]-zac_geodesic[-2]).view(-1)
    vba_z = -args.T*(zab_geodesic[-1]-zab_geodesic[-2]).view(-1)
    
    
    a_angle = torch.dot(vac_z, vab_z)/(torch.norm(vac_z)*torch.norm(vab_z))
    b_angle = torch.dot(vbc_z, vba_z)/(torch.norm(vbc_z)*torch.norm(vba_z))
    c_angle = torch.dot(vca_z, vcb_z)/(torch.norm(vca_z)*torch.norm(vcb_z))
    
    L_ab = rm.arc_length(gab_geodesic)
    L_ac = rm.arc_length(gac_geodesic)
    L_bc = rm.arc_length(gbc_geodesic)
    
    L_ab_linear = rm.arc_length(gab_linear)
    L_ac_linear = rm.arc_length(gac_linear)
    L_bc_linear = rm.arc_length(gbc_linear)
    
    a_angle = np.arccos((torch.dot(vac_z, vab_z)/(torch.norm(vac_z)*torch.norm(vab_z))).detach().numpy())
    b_angle = np.arccos((torch.dot(vbc_z, vba_z)/(torch.norm(vbc_z)*torch.norm(vba_z))).detach().numpy())
    c_angle = np.arccos((torch.dot(vca_z, vcb_z)/(torch.norm(vca_z)*torch.norm(vcb_z))).detach().numpy())
                
    save_path = args.save_path+'_same_length.pt'
    torch.save({'gab_geodesic': gab_geodesic,
                'gac_geodesic': gac_geodesic,
                'gbc_geodesic': gbc_geodesic,
                'L_ab': L_ab.item(),
                'L_ac': L_ac.item(),
                'L_bc': L_bc.item(),
                'gab_linear': gab_linear,
                'gac_linear': gac_linear,
                'gbc_linear': gbc_linear,
                'L_ab_linear': L_ab_linear.item(),
                'L_ac_linear': L_ac_linear.item(),
                'L_bc_linear': L_bc_linear.item(),
                'a_angle': a_angle,
                'b_angle': b_angle,
                'c_angle': c_angle}, 
               save_path)

    return

#%% Calling main

if __name__ == '__main__':
    main()
    
    