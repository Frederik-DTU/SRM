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

import numpy as np

#Own files
from sim_dat import sim_3d_fun

#%% Function for simulating hyper-parabolic

N_sim = 50000 #Number of simulated points
name_path = 'Data/para_data.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2-x2**2

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()

#%% Function for simulating plane (R2) in R3

N_sim = 50000 #Number of simulated points
name_path = 'Data/surface_R2.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1, x2, x1*0

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()

#%% Function for simulating parabolic

N_sim = 50000 #Number of simulated points
name_path = 'Data/parabolic.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2+x2**2

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()

#%% Function for simulating sphere-data

N_sim = 50000 #Number of simulated points
name_path = 'Data/sphere.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    N = len(x1)
    x3 = np.random.normal(0, 1, N)
    
    r = np.sqrt(x1**2+x2**2+x3**2)
    
    return x1/r, x2/r, x3/r

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()