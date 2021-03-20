# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 01:08:27 2021

@author: Frederik
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
    
def x3_fun(x1, x2):
    
    return x1**2-x2**2

class plot_3d_fun(object):
    def __init__(self,
                 fun = x3_fun,
                 N_grid = 100,
                 fig_size = (8,6)):
        
        self.fun = fun
        self.N_grid = N_grid
        self.fig_size = fig_size
        
    def convert_list_to_np(self, Z):
        
        N = len(Z)
        n = len(Z[0])
        Z_new = torch.empty(N, n)
        
        for i in range(N):
            Z_new[i] = Z[i]
            
        return Z_new.detach().numpy()
    
    def plot_geodesic_in_Z_2d(self, *args):
        
        fig = plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x, y, '-*', label=lab)
            
        plt.xlabel('t')
        plt.ylabel('')
        plt.grid()
        plt.legend()
        plt.title('Geodesic in Z')
        
        plt.tight_layout()

        
        plt.show()
        
        return
    
    def plot_dat_in_Z_2d(self, *args):
        
        fig = plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x, y, 'o', label=lab)
            
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid()
        plt.legend()
        plt.title('Z-space')
        
        plt.tight_layout()

        
        plt.show()
    
    def plot_geodesic_in_X_3d(self, x1_grid, x2_grid, *args):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N_grid)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = self.fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            z = arg[0][:,2]
            ax.plot(x, y, z, label=lab)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        
        plt.tight_layout()

        plt.show()
        
        return
        
    def true_plot_3d(self, x1_grid, x2_grid):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = self.N_grid)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = self.N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = self.fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=1.0, linewidth=0)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
                
        plt.tight_layout()
        
        plt.show()
        
    def plot_data_surface_3d(self, x1, x2, x3, title="Surface of Data"):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        ax.plot_trisurf(x1, x2, x3,
                cmap='viridis', edgecolor='none');
        ax.set_title(title);
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.tight_layout()
        
        plt.show()
        
    def plot_data_scatter_3d(self, x1, x2, x3, title="Scatter of Data"):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(min(x1), max(x1), num = self.N_grid)
        x2_grid = np.linspace(min(x2), max(x2), num = self.N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X1, X2, X3 = self.fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
                
        p = ax.scatter3D(x1, x2, x3, color='black')
        
        plt.tight_layout()
        
        plt.show()
        
    def plot_loss(self, loss, title='Loss function'):
        
        fig = plt.figure(figsize=self.fig_size)
        
        plt.plot(loss)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.title(title)
        
        plt.tight_layout()
        
        plt.show()
        
        
        

        