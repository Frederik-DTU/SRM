# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 01:08:27 2021

@author: Frederik
"""

import numpy as np
import pandas as pd 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import torch

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1**2-x2**2

class sim_plot_3d_fun(object):
    def __init__(self,
                 x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = 50000,
                 name_path = 'para_data.csv',
                 seed = 100,
                 fig_size = (8,6)):
        
        self.x1_fun = x1_fun
        self.x2_fun = x2_fun
        self.x3_fun = x3_fun
        self.N_sim = N_sim
        self.name_path = name_path
        self.seed = seed
        self.fig_size = fig_size
        
    def sim_3d(self):
    
        np.random.seed(self.seed)
        x1 = self.x1_fun(self.N_sim)
        x2 = self.x2_fun(self.N_sim)
        
        x3 = self.x3_fun(x1, x2)
        
        df = np.vstack((x1, x2, x3))
        
        pd.DataFrame(df).to_csv(self.name_path)
        
        return
    
    def plot_geodesic_in_Z_2d(self, *args):
        
        fig = plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x.detach().numpy(), y.detach().numpy(), '-*', label=lab)
            
        plt.xlabel('t')
        plt.ylabel('')
        plt.grid()
        plt.show()
        plt.legend()
        plt.tight_layout()
        plt.title('Geodesic in Z')
        
        return
        
    
    def true_plot_3d(self, x1_grid, x2_grid, N_grid, title):
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = N_grid)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X3 = self.x3_fun(X1, X2)
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        ax.plot_surface(X1, X2, X3, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
        ax.set_title(title);
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.tight_layout()
        
        plt.show()
        
    def plot_data_surface_3d(self, x1, x2, x3, title="Test"):
        
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
        
    def plot_data_scatter_3d(self, x1, x2, x3, N_grid, title="Test"):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(min(x1), max(x1), num = N_grid)
        x2_grid = np.linspace(min(x2), max(x2), num = N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X3 = self.x3_fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        
        ax.legend()
        
        p = ax.scatter3D(x1, x2, x3, color='black')
        
        plt.tight_layout()
        
        plt.show()
        
    def read_data(self):
        
        df = pd.read_csv(self.name_path, index_col=0)

        dat = torch.Tensor(df.values)
        
        dat = torch.transpose(dat, 0, 1)
        
        return dat
        
        
        
        

        