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

#%% Class to do Riemannian Computations based on data and metric matrix function

class riemannian_dgm:
    def __init__(self, T, n_encoder, n_decoder, model_encoder, model_decoder, max_iter=1000):
        self.T = T
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.max_iter = max_iter

    def interpolate(self, z0,zT):
        
        dim = z0.shape[0]
        Z = torch.empty(self.T+1, dim)
            
        for i in range(dim):
            Z[:,i] = torch.linspace(z0[i], zT[i], steps = self.T+1)
            
        return Z
    
    def energy_fun(self, G):
        
        G_sub = G[1:]-G[:-1]
        
        E = torch.sum(torch.matmul(G_sub, torch.transpose(G_sub, 0, 1)))
        E /= (2*self.T)
        
        return E
    
    def arc_length(self, G):
        
        G_sub = G[1:]-G[:-1]
        L = torch.norm(G_sub, p = 'fro')
        
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
    
    def geodesic_path_al1(self, Z, alpha = 1, eps = 0.01):
                
        grad_E = eps+1
        j = 0        
        loss = []
        Z_new = Z
                
        while (grad_E>eps and j<=self.max_iter):
            grad_E = 0.0
            G = self.model_decoder(Z_new)
            E = self.energy_fun(G)
            E.backward()
            grad = Z_new.grad.data
            grad_E = torch.sum(torch.matmul(grad, torch.transpose(grad, 0, 1)))
            
            Z_new = Z_new-alpha*grad
            
            j += 1
            loss.append(grad_E)
        
        G_old = self.model_decoder(Z)
        G_new = self.model_decoder(Z_new)
        
        L_old = self.arc_length(G_old)
        L_new = self.arc_length(G_new)
        
        return loss, Z_new, G_old, G_new, L_old, L_new
    
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
    
    
    
    
    