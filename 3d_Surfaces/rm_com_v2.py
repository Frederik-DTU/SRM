# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 01:13:39 2021

@author: Frederik
"""
# -*- coding: utf-8 -*-

import torch
from scipy.optimize import minimize
import numpy as np

#Sources:
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

class riemannian_dgm:
    def __init__(self, z0, zT, T, n_h, n_g, fun_h, fun_g, max_iter=1000):
      self.T = T
      self.fun_h = fun_h
      self.fun_g = fun_g
      self.n_h = n_h
      self.n_g = n_g
      self.z_list = self.interpolate(z0, zT)
      self.max_iter = max_iter

    
    def interpolate(self, z0,zT):
      step = (zT[0]-z0[0])/(self.T)
      a = (zT[1]-z0[1])/(zT[0]-z0[0])
      z_list = [None]*(self.T+1)
      z_list[0] = z0
      z_list[-1] = zT
      
      for i in range(1,self.T):
          zx = z_list[i-1][0]+step
          zy = i*step*a+z0[1]
          z_list[i] = torch.Tensor([zx, zy])
          
      return z_list

    def get_zlist(self):
        
        return self.z_list
        
    #The below has been modified, see source for original
    def get_jacobian(self, net_fun, x, n_out):
        x = x.squeeze()
        x = x.repeat(n_out, 1)
        x.requires_grad_(True)
        x.retain_grad()
        y = net_fun(x)
        y.backward(torch.eye(n_out), retain_graph=True)
        return x.grad.data
    
    def get_gList(self, z_list, gz0 = None, gzT = None):
    
        glist = [None]*(self.T+1)
        
        if gz0==None:
            start_idx = 0
        else:
            glist[0] = gz0
            start_idx = 1
            
        if gzT==None:
            end_idx = self.T+1
        else:
            glist[-1]=gzT
            end_idx = self.T
        
        for i in range(start_idx, end_idx):
            glist[i] = self.fun_g(z_list[i])
            
        return glist
    
    def get_list_to_torch(self, zlist, idx):
        
        z = torch.empty(self.T+1, len(idx))
        
        for i in range(0, self.T+1):
            z[i] = zlist[i][idx]
        
        return z
    
    def energy_fun(self, g_list):
        
        res = 0.0
        
        for i in range(0, self.T):
            res += torch.norm(g_list[i+1]-g_list[i])
        
        res = 1/2*self.T*res
        
        return res
    
    def geodesic_path_al1_v2(self, alpha = 1, eps = 0.01):
        
        grad_E = eps+1
        j = 0
        z_new = self.z_list[:]
        max_iter = self.max_iter
                
        gz0 = self.fun_g(z_new[0])
        gzT = self.fun_g(z_new[-1])
        
        loss = []
                
        while (grad_E>eps and j<=max_iter):
            grad_E = 0.0
            g = self.get_gList(z_new, gz0=gz0, gzT = gzT)
            for i in range(1, self.T):
                
                #Equation (5) in article:
                #jacobian_g = self.get_jacobian(self.fun_g, z_new[i], n_out=self.n_g)
                #eta_g = -self.T*torch.mv(torch.transpose(jacobian_g, 0, 1),g[i+1]-2*g[i]+g[i-1])
                
                #Equation (6) in article
                jacobian_h = self.get_jacobian(self.fun_h, g[i], n_out=self.n_h)
                eta_g = -self.T*torch.mv(jacobian_h,g[i+1]-2*g[i]+g[i-1])
                
                z_new[i] = z_new[i]-alpha*eta_g
                grad_E += torch.norm(eta_g)
            j += 1
            loss.append(grad_E)
            print(f"Iteration {j}/{max_iter} - Gradient_E: {grad_E:.8f}")
            if (j>1):
                if (loss[-1]>loss[-2]):
                    print("The Gradient is increasing. The algorithm has been Stopped! Correct alpha!")
                    break
        
        g_old = self.get_gList(self.z_list, gz0=gz0, gzT = gzT)
        g_new = self.get_gList(z_new, gz0=gz0, gzT = gzT)
        
        return loss, self.z_list, g_old, g_new, z_new
    
    def geodesic_path_al1(self, alpha = 1, eps = 0.01):
    
        grad_E = eps+1
        j = 0
        z_new = self.z_list[:]
        max_iter = self.max_iter
                
        gz0 = self.fun_g(z_new[0])
        gzT = self.fun_g(z_new[-1])
        
        loss = []
                
        while (grad_E>eps and j<=max_iter):
            grad_E = 0.0
            g = self.get_gList(z_new, gz0=gz0, gzT = gzT)
            for i in range(1, self.T):
                
                #Equation (5) in article:
                #jacobian_g = self.get_jacobian(self.fun_g, z_new[i], n_out=self.n_g)
                #eta_g = -self.T*torch.mv(torch.transpose(jacobian_g, 0, 1),g[i+1]-2*g[i]+g[i-1])
                
                #Equation (6) in article
                jacobian_h = self.get_jacobian(self.fun_h, g[i], n_out=self.n_h)
                eta_g = -self.T*torch.mv(jacobian_h,g[i+1]-2*g[i]+g[i-1])
                
                z_new[i] = z_new[i]-alpha*eta_g
                grad_E += torch.norm(eta_g)
            j += 1
            loss.append(grad_E)
            print(f"Iteration {j}/{max_iter} - Gradient_E: {grad_E:.8f}")
            if (j>1):
                if (loss[-1]>loss[-2]):
                    print("The Gradient is increasing. The algorithm has been Stopped! Correct alpha!")
                    break
        
        g_old = self.get_gList(self.z_list, gz0=gz0, gzT = gzT)
        g_new = self.get_gList(z_new, gz0=gz0, gzT = gzT)
        
        return loss, self.z_list, g_old, g_new, z_new
    
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
    
    
    
    
    
    
    
    