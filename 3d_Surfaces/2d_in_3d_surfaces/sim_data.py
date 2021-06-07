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

#%% Function for simulating hyper-parabolic

N_sim = 50000 #Number of simulated points
name_path = 'Data/hyperbolic_paraboloid.csv' #Path/file_name

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

#%% Function for simulating paraboloid

N_sim = 50000 #Number of simulated points
name_path = 'Data/paraboloid.csv' #Path/file_name

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


#%% Function for hole in paraboloid

N_sim = 50000 #Number of simulated points
name_path = 'Data/paraboloid_hole.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.uniform(-3.5,3.5, N)
            
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.uniform(-3.5,3.5, N)
            
    return x2
    
def x3_fun(x1, x2):
    
    u1 = x1
    u2 = x2
    x1 = x1[np.where( u1**2+u2**2 > 2.5 )]
    x2 = x2[np.where( u1**2+u2**2 > 2.5 )]
    
    return x1, x2, x1**2+x2**2

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()

#%% Function for non-symmetric paraboloid

N_sim = 50000 #Number of simulated points
name_path = 'Data/paraboloid_asymmetric.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    u1 = np.random.normal(1, 1, int(0.9*N))
    u2 = np.random.normal(-1, 1, int(0.1*N))
            
    return np.concatenate((u1,u2))

def x2_fun(N, mu = 0, std = 1):
    
    u1 = np.random.normal(3, 1, int(0.9*N))
    u2 = np.random.normal(-3, 1, int(0.1*N))
            
    return np.concatenate((u1,u2))
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2+x2**2

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()

#%% Function for simulating plane (R2) in R3

N_sim = 50000 #Number of simulated points
name_path = 'Data/xy_plane.csv' #Path/file_name

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

#%% Function for simulating plane (R2) in R3

N_sim = 50000 #Number of simulated points
name_path = 'Data/xy_plane_rotated.csv' #Path/file_name

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    theta = np.pi/4
    
    return x1, x2*np.cos(theta), x2*np.sin(theta)

sim = sim_3d_fun(x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = N_sim,
                 name_path = name_path)

sim.sim_3d()
