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
https://math.stackexchange.com/questions/402102/what-is-the-metric-tensor-on-the-n-sphere-hypersphere
https://stackoverflow.com/questions/32924945/finding-the-riemann-curvature-tensor-with-sympy-diffgeom-for-a-sphere
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html
https://scicomp.stackexchange.com/questions/21103/numerical-solution-of-geodesic-differential-equations-with-python
https://stackoverflow.com/questions/57532779/solution-from-scipy-solve-ivp-contains-oscillations-for-a-system-of-first-order
"""

#%% Class to do Riemannian Computations based on known metric matrix function

#Modules
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import sympy as sym

class rm_geometry:
    def __init__(self):
        
        self.G = None
        self.x = None
        self.dim = None
        self.christoffel = None
        self.bc_y0 = None
        self.bc_yT = None
        self.parallel_transport_geodesic = None
        self.v_parallel_transport_geodesic = None
        self.time_grid = None
        self.jacobian = None
        
    def compute_mmf(self, param_fun, x_sym):
        #compute metric matrix function via parametrization
        jacobian = param_fun.jacobian(x_sym)
        G = (jacobian.T)*jacobian
        
        self.jacobian = jacobian
        self.G = G
        self.x = x_sym
        self.dim = len(x_sym)
        
        return G
        
    def pass_mmf(self, G, x_sym):
        #pass metric matrix function
        self.G = G
        self.x = x_sym
        self.dim = len(x_sym)
        
        return
        
    def get_mmf(self):
        #get metric matrix function
        return self.G
    
    def get_immf(self):
        #get inverse metric matrix function 
        G_inv = (self.G).inv()
        
        return G_inv
    
    def get_christoffel_symbols(self):
        
        G_inv = np.array(self.get_immf())
        G = np.array(self.G)
        x = self.x
        
        christoffel = np.zeros((self.dim, self.dim, self.dim), dtype=object)
        
        for i in range(self.dim):
            for j in range(self.dim):
                for m in range(self.dim):
                    for l in range(self.dim):
                        christoffel[i][j][m] += (sym.diff(G[j][l],x[i])+
                                                 sym.diff(G[l][i], x[j])-
                                                 sym.diff(G[i][j], x[l]))*G_inv[l][m]
                    christoffel[i][j][m] /= 2
        
        
        return christoffel
    
    def bvp_geodesic(self, y0, yT, n_grid, y_init_grid):
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        self.bc_y0 = y0
        self.bc_yT = yT
        x_mesh = np.linspace(0,1, n_grid)
        
        sol = solve_bvp(self.__geodesic_equation_fun, 
                        self.__geodesic_equation_bc, 
                        x_mesh, y_init_grid)
        
        y = sol.y[0:self.dim]
        v = sol.y[self.dim:]
        
        return y, v
    
    def ivp_geodesic(self, n_grid, y_init):
        
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        x_mesh = np.linspace(0,1, n_grid)
        
        sol = solve_ivp(self.__geodesic_equation_fun, [0,1], y_init, t_eval=x_mesh)
        
        return sol.y[0:self.dim]
    
    def __bvp_geodesic(self, y0, yT, n_grid, y_init_grid):
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        self.bc_y0 = y0
        self.bc_yT = yT
        x_mesh = np.linspace(0,1, n_grid)
        
        sol = solve_bvp(self.__geodesic_equation_fun, 
                        self.__geodesic_equation_bc, 
                        x_mesh, y_init_grid)
        
        return sol
    
    def __ivp_geodesic(self, n_grid, y_init):
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        x_mesh = np.linspace(0,1, n_grid)
        
        sol = solve_ivp(self.__geodesic_equation_fun, [0,1], y_init, t_eval=x_mesh)
        
        return sol
    
    def __geodesic_equation_fun(self, t, y):
        
        gamma = np.array(y[0:self.dim])
        gamma_diff = np.array(y[self.dim:])
        
        chris = np.array(self.christoffel(*gamma), dtype=object)
            
        dgamma = gamma_diff
        dgamma_diff = np.zeros(gamma_diff.shape)
        
        for k in range(self.dim):
            for i in range(self.dim):
                for j in range(self.dim):
                    dgamma_diff[k] += gamma_diff[i]*gamma_diff[j]*chris[i][j][k]
            dgamma_diff[k] = -dgamma_diff[k]
    
        return np.concatenate((dgamma, dgamma_diff))
    
    def get_geodesic_equation_2d(self):
        
        gamma = sym.symbols('gamma1 gamma2')
        gamma_diff = sym.symbols('gamma_diff1 gamma_diff2')
        christoffel = self.get_christoffel_symbols()
        christoffel = sym.lambdify(self.x, christoffel, modules='numpy')
        chris = christoffel(*gamma)
        eq = []
        
        for k in range(self.dim):
            eq1 = 0.0
            for i in range(self.dim):
                for j in range(self.dim):
                    eq1 += gamma_diff[i]*gamma_diff[j]*chris[i][j][k]
                
            eq.append(sym.simplify(eq1))
    
        return eq
    
    def get_parallel_transport_equation_2d(self):
        
        v = sym.symbols('v1 v2')
        gamma = sym.symbols('gamma1 gamma2')
        gamma_diff = sym.symbols('gamma_diff1 gamma_diff2')
        
        christoffel = self.get_christoffel_symbols()
        christoffel = sym.lambdify(self.x, christoffel, modules='numpy')
        chris = christoffel(*gamma)
        
        eq = []
        for k in range(self.dim):
            eq1 = 0.0
            for i in range(self.dim):
                for j in range(self.dim):
                    eq1 += v[j]*gamma_diff[i]*chris[i][j][k]
            eq.append(sym.simplify(eq1))
        
        return eq
    
    def __geodesic_equation_bc(self, ya, yb):
        
        bc = []
        
        for i in range(self.dim):
            bc.append(ya[i]-self.bc_y0[i])
            bc.append(yb[i]-self.bc_yT[i])
            
        return bc    
    
    def num_Exp_map(self, x, v, n_grid = 100):
        
        y_init = list(x)+list(v)
        
        sol = self.__ivp_geodesic(n_grid, y_init)
        
        Exp_map = sol.y[:,-1][0:self.dim]
        
        return Exp_map
    
    def num_Log_map(self, x, y, n_grid = 100):
        
        y_init = np.zeros((2*self.dim, n_grid))
        sol = self.__bvp_geodesic(x, y, n_grid, y_init)
        v = sol.yp[:,0][0:self.dim]
        
        return v
    
    def karcher_mean_algo(self, X, mu_init = np.array([0,0,0]), tau = 0.1, eps = 0.01, 
                                                      max_iter = 100):
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        
        N = X.shape[1]
        mu = np.zeros((self.dim, max_iter+1))
        if (not mu_init.any()):
            mu[:,0] = X.mean(axis=1)
        else:
            mu[:,0] = mu_init
            
        j = 0
        while True:
            Log_sum = 0.0
            
            for i in range(N):
                Log_sum += self.__kacher_Log_map(mu[:,j], X[:,i])
            
            delta_mu = tau/N*Log_sum
            mu[:,j+1] = self.num_Exp_map(mu[:,j], delta_mu)
            
            tol = np.linalg.norm(mu[:,j+1]-mu[:,j])
            print(f"Iteration {j+1}/{max_iter} - Error: {tol:.4f}")
            if tol<eps:
                print("The Karcher Mean has succesfully been computed after j=" + str(j+1) +
                      " iterations with a tolerance of " + str(tol))
                break
            elif j+1>max_iter:
                print("The algorithm has been stopped due to the maximal number of " +
                      "iterations of " + str(max_iter) + "!")
                break
            else:
                j += 1
                
        return mu[:,0:(j+1)]
    
    def __kacher_Log_map(self, x, y, n_grid = 100):
        
        y_init = np.zeros((2*self.dim, n_grid))
        
        self.bc_y0 = x
        self.bc_yT = y
        x_mesh = np.linspace(0,1, n_grid)
        
        sol = solve_bvp(self.__geodesic_equation_fun, 
                        self.__geodesic_equation_bc, 
                        x_mesh, y_init)
        
        v = sol.yp[:,0][0:self.dim]
        
        return v
    
    def get_tangent_vector(self, y0, v0):
        
        sub = [(self.x[i], y0[i]) for i in range(len(self.x))]
        jacobian = (self.jacobian).subs(sub)
        jacobian = np.array(jacobian)
        
        u0 = v0.dot(jacobian)
        
        return u0
    
    def parallel_transport_along_geodesic(self, y0, yT, v0, n_grid):
        
        y_init_grid = np.zeros((2*self.dim , n_grid))
        sol = self.__bvp_geodesic(y0, yT, n_grid, y_init_grid)
        gamma_geodesic = sol.y[0:self.dim].transpose()
        v_geodesic = sol.y[self.dim:(2*self.dim)].transpose()
        
        n_grid = gamma_geodesic.shape[0]
        
        christoffel = self.get_christoffel_symbols()
        g_func = sym.lambdify(self.x, christoffel, modules='numpy')
        self.christoffel = g_func
        x_mesh = np.linspace(0,1, n_grid)
        
        self.time_grid = x_mesh
        self.parallel_transport_geodesic = gamma_geodesic
        self.v_parallel_transport_geodesic = v_geodesic
        sol = solve_ivp(self.__parallel_transport_geodesic_equation_fun, [0,1], v0, t_eval=x_mesh)
        
        return sol.y
    
    def arc_length(self, *args):
        
        L = 0.0
        
        if len(args)==1:
            G = args[0]
            T = G.shape[0]-1
            for i in range(T):
                g = (G[i+1]-G[i]).reshape(-1)
                L += np.sqrt(np.dot(g, g))
        elif len(args)==3:
            G = args[0]
            g0 = args[1]
            gT = args[2]
            T = G.shape[0]+1
            g = (G[0]-g0).reshape(-1)
            L += np.sqrt(np.dot(g, g))
            
            g = (gT-G[-1]).reshape(-1)
            L += np.sqrt(np.dot(g, g))
            
            for i in range(T-2):
                g = (G[i+1]-G[i]).reshape(-1)
                L += np.sqrt(np.dot(g, g))
        
        return L
    
    def geodesic_distance_matrix(self, Z, n_grid, y_init_grid, fun):
        
        N = Z.shape[0]
        dmat = np.zeros((N, N))
        
        for i in range(0, N):
            print(f"Column {i} is being computed...")
            for j in range(i+1,N):
                z_geodesic = self.bvp_geodesic(Z[i], Z[j], n_grid, y_init_grid)
                g_geodesic = fun(z_geodesic)
                L = self.arc_length(g_geodesic)
                
                dmat[i][j] = L
                dmat[j][i] = L
                
        return dmat
    
    def __parallel_transport_geodesic_equation_fun(self, t, y):
        
        
        gamma = self.parallel_transport_geodesic
        v_gamma = self.v_parallel_transport_geodesic
        t_idx = np.abs(self.time_grid - t).argmin()
        dv = np.zeros(self.dim)
        chris = np.array(self.christoffel(*gamma[t_idx,:]), dtype=object)
        
        for k in range(self.dim):
            val = 0.0
            for i in range(self.dim):
                for j in range(self.dim):
                    val += y[j]*v_gamma[t_idx,i]*chris[i][j][k]
            dv[k] = -val
        
        return dv

#%% Testing the above class                

""" 
x1, x2 = sym.symbols('x1 x2')
x = sym.Matrix([x1, x2])
param_fun = sym.Matrix([x1, x2, 0])
        
test = rm_geometry()
G = sym.simplify(test.compute_mmf(param_fun, x))
G_inv = sym.simplify(test.get_immf())
chris = sym.simplify(test.get_christoffel_symbols())
geodesic_equation = test.get_geodesic_equation_2d()
parallel_equation = test.get_parallel_transport_equation_2d()


G_inv = np.array(test.get_immf())
G = np.array(test.get_mmf())

#3,3,0
#-3,3,0
y_init = np.zeros((4, 100))

y = test.bvp_geodesic(np.array([3,3]), np.array([-3,3]), 100, y_init)
y = test.ivp_geodesic(10, [3,3,-7.49514614,   4.50945609])
v = test.parallel_transport_along_geodesic(np.array([3,3]), np.array([-3,3]), np.array([0,1]), 100)

x1 = np.linspace(-5,5,100)
x2 = np.linspace(-5,5,100)
x3 = x1**2-x2**2
X = np.vstack((x1,x2))

val = test.karcher_mean_algo(X)
"""


#%% Sources

"""
Sources:
https://towardsdatascience.com/how-to-use-pytorch-as-a-general-optimizer-a91cbf72a7fb
"""

#%% Class for Riemannian computations based on reconstructed RM using neural network

import torch
from torch import nn
from typing import List, Any
import torch.optim as optim
import numpy as np

class rm_data:
    def __init__(self,
                 model_encoder,
                 model_decoder,
                 device = 'cpu'
                 ):
        
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.device = device
        
    def interpolate(self, z0, zT, T = 100):
        
        Z = torch.empty([T+1, z0.shape[-1]])

        step = (zT-z0)/(T)
        Z[0] = z0
        for i in range(1, T):
            Z[i] = z0+i*step
        Z[-1] = zT
        
        return Z
    
    def arc_length(self, *args):
        
        L = torch.tensor(0.0)
        
        if len(args)==1:
            G = args[0]
            T = G.shape[0]-1
            for i in range(T):
                g = (G[i+1]-G[i]).view(-1)
                L += torch.sqrt(torch.dot(g, g))
        elif len(args)==3:
            G = args[0]
            g0 = args[1]
            gT = args[2]
            T = G.shape[0]+1
            g = (G[0]-g0).view(-1)
            L += torch.sqrt(torch.dot(g, g))
            
            g = (gT-G[-1]).view(-1)
            L += torch.sqrt(torch.dot(g, g))
            
            for i in range(T-2):
                g = (G[i+1]-G[i]).view(-1)
                L += torch.sqrt(torch.dot(g, g))
        
        return L
    
    def arc_length_wrong(self, *args):
        
        L = torch.tensor(0.0)
        
        if len(args)==1:
            G = args[0]
            G_dif = (G[1:]-G[0:-1]).view(-1)
            L = torch.norm(G_dif, 'fro')
        elif len(args)==3:
            G = args[0]
            g0 = args[1]
            gT = args[2]
            L += torch.norm((G[1]-g0).view(-1), 'fro')
            L += torch.norm((gT-G[-1]).view(-1), 'fro')
            
            G_dif = (G[1:]-G[0:-1]).view(-1)
            L += torch.norm(G_dif, 'fro')
        
        return L
    
    def energy_fun(self, *args):
        
        E = torch.tensor(0.0)
        
        if len(args)==1:
            G = args[0]
            G_dif = (G[1:]-G[0:-1]).view(-1)
            E += torch.norm(G_dif, 'fro')**2
        elif len(args)==3:
            G = args[0]
            g0 = args[1]
            gT = args[2]
            E += torch.norm((G[1]-g0).view(-1), 'fro')**2
            E += torch.norm((gT-G[-1]).view(-1), 'fro')**2
            
            G_dif = (G[1:]-G[0:-1]).view(-1)
            E += torch.norm(G_dif, 'fro')**2

        #Removed since there is a multiplication with the step size
        E /= 2
        E *= self.T
        
        return E
    
    def Log_map(self, z0, zT, epochs=10000, lr=1e-3, print_com = True, 
                save_step = 100, eps = 1e-6, T = 100):
        
        z_init = self.interpolate(z0, zT, T)
        _, z_geodesic = self.compute_geodesic(z_init, epochs, lr, print_com, save_step,
                                           eps)
        g_geodesic = self.model_decoder(z_geodesic)
        v_g = (g_geodesic[1]-g_geodesic[0])*T
        
        shape = list(g_geodesic.shape)
        shape = [1]+shape[1:]
        x0 = g_geodesic[0].clone().detach().requires_grad_(True).view(shape)
        z0 = self.model_encoder(x0).view(-1)
        jacobi_h = self.jacobian_mat(z0, x0)
        jacobi_h = jacobi_h.view(len(z0.view(-1)), len(x0.view(-1)))
        v_z = torch.mv(jacobi_h, v_g.view(-1))
        
        return v_z, v_g, z_geodesic, g_geodesic
    
    def get_tangent_vector(self, u0, x0):
        
        shape = list(x0.shape)
        shape = [1]+shape[1:]
        x0 = x0.view(shape)
        z0 = self.model_encoder(x0).view(-1)
        jacobi_h = self.jacobian_mat(z0, x0)
        jacobi_h = jacobi_h.view(len(z0.view(-1)), len(x0.view(-1)))
        v_z = torch.mv(jacobi_h, u0.view(-1))
        
        return v_z
    
    def linear_distance_matrix(self, Z, T=100):
        
        N = Z.shape[0]
        dmat = torch.zeros(N, N)
        
        for i in range(0, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                z_linear = self.interpolate(Z[i], Z[j], T)
                L = self.arc_length(self.model_decoder(z_linear))
                
                dmat[i][j] = L.item()
                dmat[j][i] = L.item()
                
        return dmat
    
    def euclidean_distance_matrix(self, X):
        
        N = X.shape[0]
        dmat = torch.zeros(N, N)
        
        for i in range(0, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                L = torch.sqrt(torch.dot(X[i]-X[j],X[i]-X[j]))
                
                dmat[i][j] = L.item()
                dmat[j][i] = L.item()
                
        return dmat
    
    def geodesic_distance_matrix(self, Z, epochs = 100000, lr = 1e-3, T=100):
        
        N = Z.shape[0]
        dmat = torch.zeros(N, N)
        
        for i in range(0, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                Z_int = self.interpolate(Z[i], Z[j], T)
                _, geodesic_z = self.compute_geodesic(Z_int, 
                                                      epochs = epochs,
                                                      lr = lr,
                                                      print_com = False)
                L = self.arc_length(self.model_decoder(geodesic_z))
                
                dmat[i][j] = L.item()
                dmat[j][i] = L.item()
                
        return dmat
    
    def geodesic_distance_matrix_hpc(self, Z, save_path, dmat, iter_start,
                                     epochs = 100000, lr = 1e-3, T=100):
        
        N = Z.shape[0]
        
        for i in range(iter_start, N):
            print(f"Computing row {i+1}/{N}...")
            for j in range(i+1,N):
                Z_int = self.interpolate(Z[i], Z[j], T)
                _, geodesic_z = self.compute_geodesic(Z_int, 
                                                      epochs = epochs,
                                                      lr = lr,
                                                      print_com = False)
                L = self.arc_length(self.model_decoder(geodesic_z))
                
                dmat[i][j] = L.item()
                dmat[j][i] = L.item()
            torch.save({'z_batch': Z,
                        'dmat': dmat,
                        'iter_start': i}, 
                       save_path)
                
        return dmat
        
    def compute_geodesic(self, z_init, epochs=100000, lr=1e-3, print_com = True, 
                         save_step = 100, eps = 1e-6):
        
        T = len(z_init)-1
        z0 = (z_init[0].view(1,-1)).detach()
        zT = (z_init[-1].view(1,-1)).detach()
        geodesic_z = z_init[1:T]
        geodesic_z = geodesic_z.clone().detach().requires_grad_(True)
        
        model = geodesic_path_al1(z0, zT, geodesic_z, 
                                  self.model_decoder, T).to(self.device) #Model used

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        
        loss = []
        E_prev = torch.tensor(0.0)
        for epoch in range(epochs):
            E = model()
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
            
            if np.abs(E.item()-E_prev)<eps:
                loss.append(E.item())
                break
            else:
                E_prev = E.item()
            
            if (epoch+1) % save_step == 0:
                loss.append(E.item())
                if print_com:
                    print(f"Iteration {epoch+1}/{epochs} - E_fun={E.item():.4f}")

        
        for name, param in model.named_parameters():
            geodesic_z_new = param.data
        
        geodesic_z_new = torch.cat((z0.view(1,-1), geodesic_z_new, zT.view(1,-1)), dim = 0)
        
        return loss, geodesic_z_new
    
    def compute_geodesic_fast(self, z_init, epochs=10000, lr=1e-4, eps=1e-6):
        
        T = len(z_init)-1
        z0 = (z_init[0].view(1,-1)).detach()
        zT = (z_init[-1].view(1,-1)).detach()
        geodesic_z = z_init[1:T]
        geodesic_z = geodesic_z.clone().detach().requires_grad_(True)
        model = geodesic_path_al1(z0, zT, geodesic_z, 
                                  self.model_decoder, T).to(self.device) #Model used

        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        E_prev = torch.tensor(0.0)
        for epoch in range(epochs):
            E = model()
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
            if np.abs(E.item()-E_prev)<eps:
                break
            else:
                E_prev = E.item()

        
        for name, param in model.named_parameters():
            geodesic_z_new = param.data
                
        return geodesic_z_new
    
    def compute_euclidean_mean(self, Z):
        
        mu_z = (torch.mean(Z, dim = 0)).view(1,-1)
        mu_g = self.model_decoder(mu_z)
        
        return mu_z, mu_g
    
    def compute_frechet_mean(self, X, mu_init, T=100, 
                              epochs_geodesic = 100000, epochs_frechet = 100000,
                              geodesic_lr = 1e-3, frechet_lr = 1e-3,
                              print_com = True, save_step = 1,
                              eps=1e-6):
        
        mu_init = mu_init.clone().detach().requires_grad_(True)
        model = frechet_mean(mu_init, self.model_encoder, self.model_decoder, T,
                             geodesic_epochs=epochs_geodesic,
                             geodesic_lr = geodesic_lr,
                             device = self.device).to(self.device) #Model used

        optimizer = optim.Adam(model.parameters(), lr=frechet_lr)
        
        loss = []
        L_prev = torch.tensor(0.0)
        for epoch in range(epochs_frechet):
            L = model(X)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            if np.abs(L.item()-L_prev)<eps:
                loss.append(L.item())
                break
            else:
                L_prev = L.item()
                
            if (epoch+1) % save_step == 0:
                loss.append(L.item())
                if print_com:
                    print(f"Iteration {epoch+1}/{epochs_frechet} - L={L.item():.4f}")

        
        for name, param in model.named_parameters():
            mu = param.data
                
        return loss, mu
    
    def compute_frechet_mean_hpc(self, X, mu_init, save_path, T=100, 
                              epochs_geodesic = 100000, epochs_frechet = 100000,
                              geodesic_lr = 1e-3, frechet_lr = 1e-3,
                              print_com = True, save_step = 1,
                              eps=1e-6):
        
        mu_init = mu_init.clone().detach().requires_grad_(True)
        model = frechet_mean(mu_init, self.model_encoder, self.model_decoder, T,
                             geodesic_epochs=epochs_geodesic,
                             geodesic_lr = geodesic_lr,
                             device = self.device).to(self.device) #Model used

        optimizer = optim.Adam(model.parameters(), lr=frechet_lr)
        
        loss = []
        L_prev = torch.tensor(0.0)
        
        muz_linear = mu_init
        mug_linear = self.model_decoder(muz_linear)
        for epoch in range(epochs_frechet):
            L = model(X)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            if np.abs(L.item()-L_prev)<eps:
                loss.append(L.item())
                break
            else:
                L_prev = L.item()
                
            if (epoch+1) % save_step == 0:
                for name, param in model.named_parameters():
                    muz_geodesic = param.data
                mug_geodesic = self.model_decoder(muz_geodesic)
                loss.append(L.item())
                torch.save({'loss': loss,
                             'muz_linear': muz_linear,
                             'mug_linear': mug_linear,
                             'muz_geodesic': muz_geodesic,
                             'mug_geodesic': mug_geodesic,
                             'T': T}, 
                            save_path)
                if print_com:
                    print(f"Iteration {epoch+1}/{epochs_frechet} - L={L.item():.4f}")

        
        for name, param in model.named_parameters():
            mu = param.data
                
        return loss, mu
    
    def parallel_translation_al2(self, Z, v0):
        
        T = Z.shape[0]-1
                
        #jacobi_g = self.get_jacobian(self.model_decoder, Z[0], n_decoder)
        z0 = Z[0].clone().detach().requires_grad_(True)
        y = self.model_decoder(z0.view(1,-1)).view(-1)
        jacobi_g = self.jacobian_mat(y, z0)
        u0 = torch.mv(jacobi_g, v0.view(-1))
        
        u_prev = u0
        for i in range(0,T):
            print(f"Iteration {i+1}/{T}")
            #xi = g[i] #is done in g for all
            zi = Z[i+1].clone().detach().requires_grad_(True)
            gi = self.model_decoder(zi.view(1,-1)).view(-1)
            jacobi_g = self.jacobian_mat(gi, zi)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u_prev)
            ui = torch.norm(u_prev)/torch.norm(ui)*ui
            u_prev = ui
        
        xT = self.model_decoder(Z[-1].view(1,-1)).detach().requires_grad_(True)
        zT = self.model_encoder(xT).view(-1)
        jacobi_h = self.jacobian_mat(zT, xT)
        jacobi_h = jacobi_h.view(len(zT.view(-1)), len(xT.view(-1)))
        vT = torch.mv(jacobi_h, ui)
        uT = ui
    
        return vT, uT
    
    def geodesic_shooting_al3(self, z0, u0, T=10):
        
        delta = 1/T #1/T
        x0 = self.model_decoder(z0)
        shape = x0.shape
        zdim = [T+1]
        zdim = zdim + list((z0.squeeze()).shape)
        gdim = [T+1]
        gdim = gdim + list((x0.squeeze()).shape)
        
        Z = torch.zeros(zdim)
        G = torch.zeros(gdim)
        gdim = x0.squeeze().shape
            
        zi = z0
        xi = x0.view(-1)
        u_prev = u0.view(-1)
        Z[0] = z0
        G[0] = x0
        for i in range(0, T-1):
            print(f"Iteration {i+1}/{T}")
            xi = (xi+delta*u_prev).view(shape)
            zi = self.model_encoder(xi).view(-1)
            xi = self.model_decoder(zi.view(1,-1)).view(-1)
            jacobi_g = self.jacobian_mat(xi, zi)
            U,S,V = torch.svd(jacobi_g)
            ui = torch.mv(torch.matmul(U, torch.transpose(U, 0, 1)),u_prev.view(-1))
            ui = torch.norm(u_prev)/torch.norm(ui)*ui
            u_prev = ui
            Z[i+1] = zi
            G[i+1] = xi.view(gdim)
        
        print(f"Iteration {T}/{T}")
        xT = (xi+delta*u_prev).view(shape)
        zT = self.model_encoder(xT)
        xT = self.model_decoder(zT.view(1,-1))
        Z[-1] = zT.squeeze()
        G[-1] = xT.squeeze()
        
        return Z, G, ui
    
    def linear_parallel_translation(self, za, zb, zc, T=10):
        
        shape = [T+1]+list(za.squeeze().shape)
        z_linear = torch.zeros(shape)
        v = zb-za
        step = torch.linspace(0,1,T+1)
        for i in range(T+1):
            z_linear[i] = zc+step[i]*v
            
        g_linear = self.model_decoder(z_linear)
        
        return z_linear, g_linear
        
        
    def get_jacobian(self, net_fun, x, n_out):
        x = x.squeeze()
        n = x.shape[0]
        x = x.repeat(n_out,1)
        x.requires_grad_(True)
        x.retain_grad()
        y = net_fun(x).view(n,-1)
        y.backward(torch.eye(n_out), retain_graph=True)
        return x.grad.data
    
    def jacobian_mat(self, y, x, create_graph=False):
        jac = []
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)
        for i in range(len(flat_y)):
            grad_y[i] = 1.0
            grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True,
                                          create_graph=create_graph)
            jac.append(grad_x.reshape(x.shape))
            grad_y[i] = 0.0
            
        return torch.stack(jac).reshape(y.shape+x.shape)
    
    def compute_R2_mat(self, dmat, *args):
        
        sum_idx = 0.0
        sum_total = 0.0
        
        for arg in args:
            idx = arg
            for i in idx:
                for j in idx:
                    sum_idx += dmat[i,j]
        
        N = dmat.shape[0]
        for i in range(N):
            for j in range(N):
                sum_total += dmat[i,j]
        
        R2 = 1-sum_idx/sum_total
        
        return R2, sum_total, sum_idx
    
    def compute_R2_mat_alternative(self, dmat, *args):
        
        sum_idx = 0.0
        sum_total = 0.0
        
        for arg in args:
            idx = arg
            for i in idx:
                for j in idx[i+1:]:
                    sum_idx += dmat[i,j]
        
        N = dmat.shape[0]
        for i in range(N):
            for j in range(i+1,N):
                sum_total += dmat[i,j]
        
        R2 = 1-sum_idx/sum_total
        
        return R2, sum_total, sum_idx
                

class geodesic_path_al1(nn.Module):
    def __init__(self,
                 z0,
                 zT,
                 geodesic_z,
                 model_decoder,
                 T
                 ):
        super(geodesic_path_al1, self).__init__()
    
        self.geodesic_z = nn.Parameter(geodesic_z, requires_grad=True)
        self.model_decoder = model_decoder
        self.g0 = (model_decoder(z0)).detach()
        self.gT = (model_decoder(zT)).detach()
        self.T = T
    
    def forward(self):
        
        E = torch.tensor(0.0)
        G = self.model_decoder(self.geodesic_z)
        
        g = (G[0]-self.g0).view(-1)
        E += torch.dot(g, g)
        g = (self.gT-G[-1]).view(-1)
        E += torch.dot(g, g)
        
        G_dif = G[1:]-G[0:-1]
        E += torch.norm(G_dif, 'fro')**2
        
        #Aletnative for two above lines
        #for i in range(self.T-2):
        #    g = (G[i+1]-G[i]).view(-1)
        #    E += torch.dot(g, g)
        
        #Removed since there is a multiplication with the step size
        E /= 2
        E *= self.T
        
        return E
        
class frechet_mean(nn.Module):
    def __init__(self,
                 mu_init,
                 model_encoder,
                 model_decoder,
                 T = 100,
                 geodesic_epochs = 10000,
                 geodesic_lr = 1e-5,
                 device = 'cpu'
                 ):
        super(frechet_mean, self).__init__()
            
        self.mu = nn.Parameter(mu_init, requires_grad=True)
        self.model_decoder = model_decoder
        self.T = T
        self.rm = rm_data(model_encoder, model_decoder, device)
        self.epochs = geodesic_epochs
        self.lr = geodesic_lr
        self.device = device
    
    def forward(self, z):
        
        L = torch.tensor(0.0)
        N = z.shape[0]
        
        for i in range(N):
            dat = z[i]
            z_init = self.rm.interpolate(self.mu, dat, self.T)
            g0 = self.model_decoder(self.mu.view(1,-1))
            gT = self.model_decoder(dat.view(1,-1)).detach()
            geodesic_z = self.rm.compute_geodesic_fast(z_init, epochs=self.epochs, lr = self.lr)
            geodesic_g = self.model_decoder(geodesic_z).detach()
            L += self.rm.arc_length(geodesic_g, g0, gT)**2
        
        return L
        




