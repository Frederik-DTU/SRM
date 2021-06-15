# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 00:44:58 2021

@author: Frederik
"""

#%% Sources:
    
"""
Sources:
http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
"""

#%% Modules

#Loading own module from parent folder
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.realpath(currentdir))
parentdir = os.path.dirname(os.path.realpath(parentdir))
sys.path.append(parentdir)

#Modules
import numpy as np

#Own files
from sim_dat import sim_3d_fun

#%% Function for simulating sphere-data

N_sim = 50000 #Number of simulated points
name_path = 'Data/sphere.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    theta = np.linspace(0, 2*np.pi, N)
    #x1 = np.random.normal(mu, std, N)
    
    return theta

def x2_fun(N, mu = 0, std = 1):
    
    phi = np.linspace(0, np.pi, N)
    #x2 = np.random.normal(mu, std, N)
    
    return phi
    
def x3_fun(x1, x2):
        
    return np.cos(x1)*np.sin(x2), np.sin(x1)*np.sin(x2), np.cos(x1)

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()
