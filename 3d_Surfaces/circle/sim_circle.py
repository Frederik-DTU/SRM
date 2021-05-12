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
import pandas as pd

#%% Function for simulating circle in R^3

N_sim = 50000 #Number of simulated points
name_path = 'Data/circle.csv' #Path/file_name
mean = np.array([1.,1.,1.])

def x_fun(N, mu = mean):
    
    theta = np.random.uniform(0, 2*np.pi, N)
    x1 = np.cos(theta)+mean[0]
    x2 = np.sin(theta)+mean[1]
    x3 = np.zeros(N)+mean[2]
    
    df = np.vstack((x1, x2, x3))    
    
    return df

df = x_fun(N_sim, mean)
        
pd.DataFrame(df).to_csv(name_path)