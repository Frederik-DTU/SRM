# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 01:13:39 2021

@author: Frederik
"""
# -*- coding: utf-8 -*-

#%% Sources

"""
Sources (USE THEM!!!! ESPECIALLY NUMBER ONE!!!):
https://cardona.co/math/2018/09/16/geodesics.html
https://docs.sympy.org/latest/modules/diffgeom.html
"""

#%% Modules

import numpy as np
from scipy.integrate import solve_bvp

#%% Class to do Riemannian Computations based on data and metric matrix function

class rm_computations:
    ##def __init__(self):

    def bvp_geodesic(self, y0, y1, christoffel_fun):
        
        
        
        