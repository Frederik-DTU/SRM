# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:11:39 2021

@author: Frederik
"""

import torch
from scipy.optimize import minimize
import numpy as np

#Sources:
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

class riemannian_dgm:
    def __init__(self, z0, zT, T, n_h, n_g, fun_h, fun_g):
      self.T = T
      self.fun_h = fun_h
      self.fun_g = fun_g
      self.n_h = n_h
      self.n_g = n_g
      self.z_list = self.interpolate(z0, zT)

    
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
    
    def energy_fun(self, g_list):
        
        res = 0.0
        
        for i in range(0, self.T):
            res += torch.norm(g_list[i+1]-g_list[i])
        
        res = 1/2*self.T*res
        
        return res
    
    def get_list_to_torch(self, zlist, idx):
        
        z = torch.empty(self.T+1, len(idx))
        
        for i in range(0, self.T+1):
            z[i] = zlist[i][idx]
        
        return z
    
    def obj_al1(self, alpha, g):
        
        z_path = self.z_list[:]
        grad_E = 0.0
        alpha = torch.tensor(alpha, dtype=torch.float32)
                
        for i in range(1, self.T):
                
            #Equation (5) in article:
            jacobian_g = self.get_jacobian(self.fun_g, z_path[i], n_out=self.n_g)
            eta_g = -self.T*torch.mv(torch.transpose(jacobian_g, 0, 1),g[i+1]-2*g[i]+g[i-1])
            
            #Equation (6) in article
            #jacobian_h = self.get_jacobian(self.fun_h, g[i], n_out=self.n_h)
            #eta_g = -self.T*torch.mv(jacobian_h,g[i+1]-2*g[i]+g[i-1])
            
            z_path[i] = z_path[i]-alpha*eta_g
            grad_E += torch.norm(eta_g)
        
        g_old = self.get_gList(self.z_list, gz0=g[0], gzT = g[-1])
        g_new = self.get_gList(z_path, gz0=g[0], gzT = g[-1])
        E_old = self.energy_fun(g_old)
        E_new = self.energy_fun(g_new)
    
            
        return (E_new-E_old).detach().numpy().astype(np.float64)
    
    def get_iter_al1(self, g, bnds = [(0,1)], alpha0=0.5):
        
        sol = minimize(self.obj_al1, alpha0, method = 'SLSQP', bounds = bnds, args = (g))
        
        #sol = minimize(self.obj_al1, alpha0, jac=True, method='L-BFGS-B', bounds = bnds, args = (g))
        
        return sol.x
    
    def line_search_alpha_al1(self, g, bnds = [0,1], max_steps=1000):
        
        lb = bnds[0]
        ub = bnds[1]
        alpha0 =(lb+ub)/2
        E_ub = self.obj_al1(ub, g)
        if (E_ub<0):
            return ub
        E_alpha0 = self.obj_al1(alpha0, g)
        
        for i in range(0, max_steps):
            print(alpha0)
            E_alpha0 = self.obj_al1(alpha0, g)
            
            if (E_alpha0<0):
                alpha = alpha0
                break
            else:
                ub = alpha0
                alpha0 = (lb+ub)/2
                
        return alpha
    
    def geodesic_path_al1(self, alpha = 1, eps = 0.01):
    
        grad_E = eps+1
        #grad_prev = math.inf
        #delta = 1/self.T
        z_path = self.z_list[:]
                
        gz0 = self.fun_g(z_path[0])
        gzT = self.fun_g(z_path[-1])
        
        g = self.get_gList(z_path, gz0=gz0, gzT = gzT)
        #alpha_test = self.get_iter_al1(g)
        #alpha_test = self.line_search_alpha_al1(g)
        #alpha = torch.tensor(alpha_test, dtype=torch.float32)
        #print(alpha_test)
        
        #g_old = self.get_gList(self.z_list, gz0=gz0, gzT = gzT)
        #E_old = self.energy_fun(g_old)
        #print(E_old)
                
        while grad_E>eps:
            print(grad_E)
            grad_E = 0.0
            g = self.get_gList(z_path, gz0=gz0, gzT = gzT)
            for i in range(1, self.T):
                
                #Equation (5) in article:
                #jacobian_g = self.get_jacobian(self.fun_g, z_path[i], n_out=self.n_g)
                #eta_g = -self.T*torch.mv(torch.transpose(jacobian_g, 0, 1),g[i+1]-2*g[i]+g[i-1])
                
                #Equation (6) in article
                jacobian_h = self.get_jacobian(self.fun_h, g[i], n_out=self.n_h)
                eta_g = -self.T*torch.mv(jacobian_h,g[i+1]-2*g[i]+g[i-1])
                
                z_path[i] = z_path[i]-alpha*eta_g
                grad_E += torch.norm(eta_g)
        
        g_old = self.get_gList(self.z_list, gz0=gz0, gzT = gzT)
        g_new = self.get_gList(z_path, gz0=gz0, gzT = gzT)
        E_old = self.energy_fun(g_old)
        E_new = self.energy_fun(g_new)
        
        return E_old, E_new, z_path
    
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
    
    
    
    
    
    
    
    