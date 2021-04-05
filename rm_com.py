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
                 model_encoder, 
                 model_decoder,
                 T = 10,
                 MAX_ITER=10000,
                 eps = 0.01,
                 div_fac = 2):
        
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.T = T
        self.MAX_ITER = MAX_ITER
        self.eps = eps
        self.div_fac = div_fac

    def interpolate(self, z0, zT):
        
        Z = torch.empty([self.T+1, z0.shape[-1]])

        step = (zT-z0)/(self.T)
        Z[0] = z0
        for i in range(1, self.T):
            Z[i] = z0+i*step
        Z[-1] = zT
        
        return Z
    
    def energy_fun(self, G, g0, gT):
        
        E = 0.0
        
        g = (G[0]-g0).view(-1)
        E += torch.dot(g, g)
        g = (gT-G[-1]).view(-1)
        E += torch.dot(g, g)
        
        G_dif = G[1:]-G[0:-1]
        E += torch.norm(G_dif, 'fro')**2

        #Removed since there is a multiplication with the step size
        E /= 2
        E *= self.T
        
        return E
    
    def energy_fun_mat(self, G):
        
        G_dif = G[1:]-G[0:-1]
        E = torch.norm(G_dif, 'fro')**2

        #Removed since there is a multiplication with the step size
        E /= 2
        E *= self.T
        
        return E
    
    def arc_length(self, G, g0, gT):
        
        L = 0.0
        
        L += torch.norm(G[1]-g0, 'fro')
        L += torch.norm(gT-G[-1], 'fro')
        
        G_dif = G_dif = G[1:]-G[0:-1]
        L += torch.norm(G_dif, 'fro')
        
        return L
    
    def arc_length_mat(self, G):
        
        G_dif = G_dif = G[1:]-G[0:-1]
        L = torch.norm(G_dif, 'fro')
        
        return L
    
    def get_g_fun(self, Z, g_fun):
        
        g0 = g_fun[0]
        dim = g0.shape
        dim.insert(0, self.T+1)
        
        G = torch.empty(dim)
        G[0] = g0
        
        for i in range(1, self.T+1):
            G[i] = g_fun(Z[i])
            
        return G
    
    def get_euclidean_mean(self, Z):
        
        mu_z = (torch.mean(Z, dim = 0)).view(1,-1)
        mu_g = self.model_decoder(mu_z)
        
        return mu_z, mu_g
    
    def geodesic_distance_matrix(self, Z, alpha = 0.01):
        
        N = Z.shape[0]
        dmat = torch.zeros(N, N)
        
        for i in range(0, N):
            for j in range(i+1,N):
                Z_int = self.interpolate(Z[i], Z[j])
                _, _, _, _, _, _, L_new = self.geodesic_path_al1(Z_int, alpha = alpha)
                dmat[i][j] = L_new.item()
                dmat[j][i] = L_new.item()
                
        return dmat
    
    def get_geodesic_using_metric(self, Z, g_fun, Jg_fun, alpha = 0.1, print_conv = False):
        
        grad_E = self.eps+1
        count = 0        
        loss = []
        E_fun = []
        Z_new = Z
        Z_dummy = Z
                
        G_old = self.get_g_fun(Z_new, g_fun)            
        E = self.energy_fun_mat(G_old)
        L_old = self.arc_length(G_old)
        E_fun.append(E.item())
        while (grad_E>self.eps and count<=self.MAX_ITER):
            grad_E = 0.0
            
            for i in range(1, self.T):
                jacobi = torch.transpose(Jg_fun(Z_new[i]),0,1)
                dE_dZ = -torch.matmul(jacobi, g_fun(Z_new[i+1])+g_fun(Z_new[i-1])-
                                      2*g_fun(Z_new[i]))
                
                Z_dummy[i] = Z_new[i]-alpha*dE_dZ
                
                grad_E += torch.dot(dE_dZ, dE_dZ)
            
            
            loss.append(grad_E.item())
            
            G = self.model_decoder(Z_dummy)
            E = self.energy_fun_mat(G)
            if (E.item()-E_fun[-1]>0):
                G = self.model_decoder(Z_new)
                E = self.energy_fun_mat(G)
                alpha /= self.div_fac
            else:
                E_fun.append(E.item())
                Z_new = Z_dummy
            
            if print_conv:
                print(f"Iteration {count}/{self.MAX_ITER} - E_fun={E_fun[-1]:.4f}, "
                      f"grad_E={loss[-1]:.4f} " 
                      f"(alpha={alpha:.16f})")
                
            count += 1  
        
        G_new = self.get_g_fun(Z_new, g_fun)            
        L_new = self.arc_length(G_new)
        
        if print_conv:
            if grad_E<self.eps:
                print("The geodesic has converged!")
            else:
                print("The algorithm stopped due to maximum number of iterations!")
        
        return loss, E_fun, Z_new, G_old, G_new, L_old, L_new
    
    def get_frechet_mean(self, Z, alpha_mu = 0.1, alpha_g = 0.1):
        
        count = 0
        L = []
        step = self.eps + 1
        N = Z.shape[0]
        
        muz_init, mug_init = self.get_euclidean_mean(Z)
        mu_z = muz_init
        mu_dummy = mu_z
        
        while (step>self.eps and count<=self.MAX_ITER):
            
            L_val = 0.0
            for i in range(N):
                Z_int = self.interpolate(mu_z, Z[i])
                _, _, _, _, _, _, L_new = self.geodesic_path_al1(Z_int, alpha = alpha_g)
                L_val += L_new
                
            dL = (grad(outputs = L_val, inputs = mu_z)[0]).view(-1)
            
            mu_dummy = mu_z-alpha_mu*dL
            step = torch.dot(dL, dL)
                        
            if count>1:
                if (L[-1]-L[-2]>0):
                    L.append(L[-1])
                    alpha_mu /= self.div_fac
                else:
                    L.append(L_val.item())
                    mu_z = mu_dummy
            else:
                L.append(L_val.item())
                mu_z = mu_dummy

            count += 1

            print(f"Iteration {count}/{self.MAX_ITER} - Gradient_Step: {step:.4f}\t" 
                  f"TOLERANCE={self.eps:.4f} (alpha_mu={alpha_mu:.8f})")
                    
        mu_g = self.model_decoder(mu_z)
        
        return L, muz_init, mug_init, mu_z, mu_g
    
    def geodesic_path_al1(self, Z, alpha = 0.1, print_conv = False):
        
        grad_E = self.eps+1
        count = 0        
        loss = []
        E_fun = []
        Z_new = Z[1:self.T]
        Z_dummy = Z_new
        
        G_old = self.model_decoder(Z)
        g0 = G_old[0].detach()
        gT = G_old[-1].detach()
        L_old = self.arc_length(G_old, g0, gT)
        
        G = self.model_decoder(Z_dummy)
        E = self.energy_fun(G, g0, gT)
        E_fun.append(E.item())
        
        while (grad_E>self.eps and count<=self.MAX_ITER):
    
            dE_dZ =(grad(outputs = E, inputs = Z_new)[0])
            Z_dummy = Z_new-alpha*dE_dZ
            grad_E = torch.linalg.norm(dE_dZ, 'fro')**2
            
            loss.append(grad_E.item())
            
            
            G = self.model_decoder(Z_dummy)
            E = self.energy_fun(G, g0, gT)
            if (E.item()-E_fun[-1]>0):
                G = self.model_decoder(Z_new)
                E = self.energy_fun(G, g0, gT)
                alpha /= self.div_fac
            else:
                E_fun.append(E.item())
                Z_new = Z_dummy
            
            if print_conv:
                print(f"Iteration {count}/{self.MAX_ITER} - E_fun={E_fun[-1]:.4f}, "
                      f"grad_E={loss[-1]:.4f} " 
                      f"(alpha={alpha:.16f})")
                
            count += 1  
        
        G_new = self.model_decoder(Z_new)
        L_new = self.arc_length(G_new, g0, gT)
        
        if print_conv:
            if grad_E<self.eps:
                print("The geodesic has converged!")
            else:
                print("The algorithm stopped due to maximum number of iterations!")
        
        return loss, E_fun, Z_new, G_old, G_new, L_old, L_new
    
    def parallel_translation_al2(self, Z, v0):
        
        jacobi_dim = len(v0)
        
        jacobi_g = self.get_jacobian(self.model_decoder, Z[0], jacobi_dim)
        u0 = torch.mv(jacobi_g, v0)
        
        for i in range(0,self.T):
            #xi = g[i] #is done in g for all
            jacobi_g = self.get_jacobian(self.model_decoder, Z[i+1], jacobi_dim)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u0)
            ui = torch.norm(u0)/torch.norm(ui)*ui
        
        xT = self.model_decoder(Z[-1])
        jacobi_h = self.get_jacobian(self.model_encoder, xT, jacobi_dim)    
        vT = torch.mv(jacobi_h, ui)
    
        return vT
    
    def geodesic_shooting_al3(self, x0, z0, u0):
    
        delta = 1/self.T
        jacobi_dim = len(u0)
    
        zi = z0.view(1,-1)
        xi = x0.view(1,-1)
        for i in range(0, self.T):
            xi = (xi+delta*u0).view(1,-1)
            zi = self.model_encoder(xi)
            xi = self.model_decoder(zi)
            jacobi_g = self.get_jacobian(self.model_decoder, zi, jacobi_dim)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u0)
            ui = torch.norm(u0)/torch.norm(ui)*ui
            u0 = ui
            
        return zi   
    
    def get_jacobian(self, net_fun, x, n_out):
        x = x.squeeze()
        x = x.repeat(n_out,1)
        x.requires_grad_(True)
        x.retain_grad()
        y = net_fun(x)
        y.backward(torch.eye(n_out), retain_graph=True)
        return x.grad.data

#%% The old and most likely inefficient way of computing jacobi matrix

"""
    #The below has been modified, see source for original
    def get_jacobian(self, net_fun, x, n_out):
        x = x.squeeze()
        x = x.repeat(n_out, 1)
        x.requires_grad_(True)
        x.retain_grad()
        y = net_fun(x)
        y.backward(torch.eye(n_out), retain_graph=True)
        return x.grad.data
"""
