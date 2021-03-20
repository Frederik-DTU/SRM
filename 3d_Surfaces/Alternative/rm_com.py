# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 01:13:39 2021

@author: Frederik
"""
# -*- coding: utf-8 -*-

#%% Sources

"""
Sources:
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
"""

#%% Modules

import torch
from torch.autograd import grad

#%% Class to do Riemannian Computations based on data and metric matrix function

class riemannian_data:
    def __init__(self, 
                 T, 
                 h_dim, 
                 g_dim, 
                 model_encoder, 
                 model_decoder, 
                 max_iter=1000,
                 eps = 0.01):
        self.T = T
        self.h_dim = h_dim
        self.g_dim = g_dim
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.max_iter = max_iter
        self.eps = eps

    def interpolate(self, z0, zT):
        
        Z = [None]*(self.T+1)
        Z[0] = z0
        Z[-1] = zT
        step = (zT-z0)/(self.T)
        
        for i in range(1, self.T):
            Z[i] = Z[i-1]+step
            
        return Z
    
    def get_decoded(self, Z):
        
        G = [None]*(self.T+1)
        
        for i in range(self.T+1):
            G[i] = self.model_decoder(Z[i])
            
        return G
    
    def energy_fun(self, G):
        
        E = 0.0
        for i in range(self.T):
            g = G[i+1]-G[i]
            E += torch.dot(g, g)

        #Removed since there is a multiplication with the step size
        #E /= 2
        #E *= self.T
        
        return E
    
    def arc_length(self, G):
        
        L = 0.0
        for i in range(self.T):
            L += torch.norm(G[i+1]-G[i], 'fro')
        
        return L
    
    #The below has been modified, see source for original
    def get_jacobian(self, net_fun, x, n_out):
        x = x.squeeze()
        x = x.repeat(n_out, 1)
        x.requires_grad_(True)
        x.retain_grad()
        y = net_fun(x)
        y.backward(torch.eye(n_out), retain_graph=True)
        return x.grad.data
    
    def get_euclidean_mean(self, Z):
        
        mu_z = torch.mean(Z, dim = 0)
        mu_g = self.model_decoder(mu_z)
        
        return mu_z, mu_g
    
    def get_geodesic_distance_to_point(self, mu, Z, alpha = 0.01):
        
        n = len(Z)
        L = 0.0
        
        for i in range(n):
            Z_int = self.interpolate(mu, Z[i])
            _, _, _, _, _, _, L_new = self.geodesic_path_al1(Z_int, alpha = alpha)
            L += L_new
            
        dL = grad(outputs = L, inputs = mu)[0]
            
        return L, dL
    
    def get_frechet_mean(self, Z, alpha_mu = 0.01, alpha_d = 0.01):
        
        muz_init, mug_init = self.get_euclidean_mean(Z)
        j = 0
        L = []
        step = self.eps+1
        mu = muz_init
        
        while (step>self.eps and j<=self.max_iter):
            
            L_val, dL = self.get_geodesic_distance_to_point(mu, Z, alpha_d)
            print(j)
            L.append(L_val)
            mu = mu-alpha_mu*dL
            step = torch.dot(dL, dL)
            print(step)
            
            j += 1
        
        mu_z = mu
        mu_g = self.model_decoder(mu_z)
        
        return L, muz_init, mug_init, mu_z, mu_g
    
    def geodesic_path_al1(self, Z, alpha = 0.01, print_conv = False):
                
        grad_E = self.eps+1
        j = 0        
        loss = []
        E_fun = []
        Z_new = Z[:]
                
        while (grad_E>self.eps and j<=self.max_iter):
            grad_E = 0.0
            G = self.get_decoded(Z_new)
            
            E = self.energy_fun(G)
            for i in range(1, self.T):
                dE_dZ = grad(outputs = E, inputs = Z_new[i], retain_graph=True)[0]
                
                Z_new[i] = Z_new[i]-alpha*dE_dZ
                
                grad_E += torch.dot(dE_dZ, dE_dZ)
            
            E_fun.append(E)
            loss.append(grad_E)
            j += 1        
        
        G_old = self.get_decoded(Z)
        L_old = self.arc_length(G_old)
        
        G_new = self.get_decoded(Z_new)
        L_new = self.arc_length(G_new)
        
        if print_conv:
            if grad_E>self.eps:
                print("The geodesic has converged!")
            else:
                print("The algorithm stopped due to maximum number of iterations!")
        
        return loss, E_fun, Z_new, G_old, G_new, L_old, L_new
    
    def parallel_translation_al2(self, z_list, v0):
        
        u0 = torch.mv(self.get_jacobian(self.fun_g, z_list[0], self.n_g), v0)
        g = self.get_gList(z_list)
        
        for i in range(0,self.T):
            #xi = g[i] #is done in g for all
            jacobian_g = self.get_jacobian(self.fun_g, z_list[i+1], self.n_g)
            U,S,V = torch.svd(jacobian_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u0)
            ui = torch.norm(u0)/torch.norm(ui)*ui
            u0 = ui
            
        jacobian_h = self.get_jacobian(self.fun_h, g[-1], self.n_h)        
        vT = torch.mv(jacobian_h, ui)
    
        return vT
    
    def geodesic_shooting_al3(self, x0, z0, u0):
    
        delta = 1/self.T
    
        zi = z0
        xi = x0
        for i in range(0, self.T):
            xi = xi+delta*u0
            zi = self.fun_h(xi)
            xi = self.fun_g(zi)
            jacobian_g = self.get_jacobian(self.fun_g, zi, self.n_g)
            U,S,V = torch.svd(jacobian_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u0)
            ui = torch.norm(u0)/torch.norm(ui)*ui
            u0 = ui
            
        return zi
    
    
def energy_fun(G):
        
    G_sub = G[1:]-G[:-1]
    
    E = torch.sum(torch.matmul(G_sub, torch.transpose(G_sub, 0, 1)))
    E /= (2*10)
    
    return E

#%%