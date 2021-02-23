# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:18:47 2021

@author: Frederik
"""

#%% Sources

"""
Sources used:
https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb
"""


#%% Modules

import torch
from torch import nn
from typing import List, Any

#%% VAE for 3d surfaces using article network

class VAE_3d(nn.Module):
    def __init__(self,
                 fc_h: List[int] = [3,100],
                 fc_g: List[int] = [2, 100, 3],
                 fc_mu: List[int] = [100, 2],
                 fc_var: List[int] = [100, 2],
                 fc_h_act: List[Any] = [nn.ELU],
                 fc_g_act: List[Any] = [nn.ELU, nn.Identity],
                 fc_mu_act: List[Any] = [nn.Identity],
                 fc_var_act: List[Any] = [nn.Sigmoid]
                 ):
        super(VAE_3d, self).__init__()
    
        self.fc_h = fc_h
        self.fc_g = fc_g
        self.fc_mu = fc_mu
        self.fc_var = fc_var
        self.fc_h_act = fc_h_act
        self.fc_g_act = fc_g_act
        self.fc_mu_act = fc_mu_act
        self.fc_var_act = fc_var_act
        
        self.num_fc_h = len(fc_h)
        self.num_fc_g = len(fc_g)
        self.num_fc_mu = len(fc_h)
        self.num_fc_var = len(fc_h)
        
        self.h = self.encoder()
        self.g = self.decoder()
        self.mu_net = self.mu_layer()
        self.var_net = self.var_layer()
        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
    
    def encoder(self):
        
        layer = []
        
        for i in range(1, self.num_fc_h):
            layer.append(nn.Linear(self.fc_h[i-1], self.fc_h[i]))
            layer.append(self.fc_h_act[i-1]())
            #input_layer.append(self.activations_h[i](inplace=True))
            
        return nn.Sequential(*layer)
    
    def mu_layer(self):
        
        layer = []
        
        for i in range(1, self.num_fc_mu):
            layer.append(nn.Linear(self.fc_mu[i-1], self.fc_mu[i]))
            layer.append(self.fc_mu_act[i-1]())
            
        return nn.Sequential(*layer)
    
    def var_layer(self):
        
        layer = []
        
        for i in range(1, self.num_fc_var):
            layer.append(nn.Linear(self.fc_var[i-1], self.fc_var[i]))
            layer.append(self.fc_var_act[i-1]())
            
        return nn.Sequential(*layer)
    
    def rep_par(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mu + (eps * std)
        return z
        
    def decoder(self):
        
        layer = []
        
        for i in range(1, self.num_fc_g):
            layer.append(nn.Linear(self.fc_g[i-1], self.fc_g[i]))
            layer.append(self.fc_g_act[i-1]())
            
        return nn.Sequential(*layer)
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        
        return log_pxz.sum(dim=1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        
        return kl
    
    def forward(self, x):
        
        x_encoded = self.h(x)
        mu, var = self.mu_net(x_encoded), self.var_net(x_encoded)
        z = self.rep_par(mu, var)
        x_hat = self.g(z)
        
        std = torch.sqrt(var)
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kld = self.kl_divergence(z, mu, std)
        rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        
        # elbo
        elbo = (kld - rec_loss)
        elbo = elbo.mean()
        
        return z, x_hat, mu, var, kld.mean(), -rec_loss.mean(), elbo

#%% Simple test
        
#Simple test
#vae = VAE_3d()
#x = torch.tensor([[1,2,3], [4,5,6], [7, 8, 9]], dtype=torch.torch.float)
#z, x_hat, mu, var, kld, rec_loss, elbo = vae(x)
        
#%% Alternative implementation (Should give the same)

#The training script should be modified for the version below.
class VAE_3d_non_prob(nn.Module):
    def __init__(self,
                 fc_h: List[int] = [3,100],
                 fc_g: List[int] = [2, 100, 3],
                 fc_mu: List[int] = [100, 2],
                 fc_var: List[int] = [100, 2],
                 fc_h_act: List[Any] = [nn.ELU],
                 fc_g_act: List[Any] = [nn.ELU, nn.Identity],
                 fc_mu_act: List[Any] = [nn.Identity],
                 fc_var_act: List[Any] = [nn.Sigmoid],
                 ):
        super(VAE_3d_non_prob, self).__init__()
    
        self.fc_h = fc_h
        self.fc_g = fc_g
        self.fc_mu = fc_mu
        self.fc_var = fc_var
        self.fc_h_act = fc_h_act
        self.fc_g_act = fc_g_act
        self.fc_mu_act = fc_mu_act
        self.fc_var_act = fc_var_act
        
        self.num_fc_h = len(fc_h)
        self.num_fc_g = len(fc_g)
        self.num_fc_mu = len(fc_h)
        self.num_fc_var = len(fc_h)
        
        self.h = self.encoder()
        self.g = self.decoder()
        self.mu_net = self.mu_layer()
        self.var_net = self.var_layer()
        
        
    def encoder(self):
        
        layer = []
        
        for i in range(1, self.num_fc_h):
            layer.append(nn.Linear(self.fc_h[i-1], self.fc_h[i]))
            layer.append(self.fc_h_act[i-1]())
            #input_layer.append(self.activations_h[i](inplace=True))
            
        return nn.Sequential(*layer)
    
    def mu_layer(self):
        
        layer = []
        
        for i in range(1, self.num_fc_mu):
            layer.append(nn.Linear(self.fc_mu[i-1], self.fc_mu[i]))
            layer.append(self.fc_mu_act[i-1]())
            
        return nn.Sequential(*layer)
    
    def var_layer(self):
        
        layer = []
        
        for i in range(1, self.num_fc_var):
            layer.append(nn.Linear(self.fc_var[i-1], self.fc_var[i]))
            layer.append(self.fc_var_act[i-1]())
            
        return nn.Sequential(*layer)
        
    def rep_par(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        z = mu + (std*eps)
        return z
        
    def decoder(self):
        
        layer = []
        
        for i in range(1, self.num_fc_g):
            layer.append(nn.Linear(self.fc_g[i-1], self.fc_g[i]))
            layer.append(self.fc_g_act[i-1]())
            
        return nn.Sequential(*layer)
        
    def forward(self, x):
        
        x = self.h(x)
        mu = self.mu_net(x)
        var = self.var_net(x)
        z = self.rep_par(mu, var)
        x_rec = self.g(z)
        
        return z, x_rec, mu, var








        