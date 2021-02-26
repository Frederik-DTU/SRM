# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:42:09 2021

@author: Frederik
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable



#Sources:
#https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
#https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb
#https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html

class vae_para_test(nn.Module):
    
    def __init__(self, device = 'cuda'):
        super(vae_para_test, self).__init__()
        
        self.fch1 = nn.Linear(3, 100)        
        
        self.fcmu = nn.Linear(100, 2)
        self.fcvar = nn.Linear(100,2)
        
        self.fcg1 = nn.Linear(2,100)
        self.fcg2 = nn.Linear(100,3)
        self.device = device
    
    def encoder(self, x):
    
        x = F.elu(self.fch1(x))
        mu = self.fcmu(x)
        var = torch.sigmoid(self.fcvar(x))
        
        return mu, var
        
    def rep_par(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return mu+eps*std
        
    def decoder(self, z):
        
        z = F.elu(self.fcg1(z))
        z = self.fcg2(z)
        
        return z
        
    def forward(self, x):
        
        mu, var = self.encoder(x)
        z = self.rep_par(mu, var)
        x_rec = self.decoder(z)
        
        return z, x_rec, mu, var
    
    def get_decoded(self, z):
        
        x_rec = self.decocder(z)
        
        return x_rec
    
    def get_encoded(self, x):
        
        mu, var = self.encoder(x)
        
        return mu

class VAE_prob(nn.Module):
    def __init__(self, device = 'cuda'):
        super(VAE_prob, self).__init__()
        
        self.fch1 = nn.Linear(3, 100)        
        
        self.fcmu = nn.Linear(100, 2)
        self.fcvar = nn.Linear(100,2)
        
        self.fcg1 = nn.Linear(2,100)
        self.fcg2 = nn.Linear(100,3)
        self.device = device

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
    def encoder(self, x):
    
        x = F.elu(self.fch1(x))
        mu = self.fcmu(x)
        var = torch.sigmoid(self.fcvar(x))
        
        return mu, var
        
    def rep_par(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.randn_like(std).to(self.device)
        return mu+eps*std
        
    def decoder(self, z):
        
        z = F.elu(self.fcg1(z))
        z = self.fcg2(z)
        
        return z

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
        
        mu, var = self.encoder(x)
        
        std = torch.sqrt(var)
        z = self.rep_par(mu, var)
        
        x_hat = self.decoder(z)
        
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        
        # kl
        kl = self.kl_divergence(z, mu, std)
        
        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()
        
        return z, x_hat, mu, var, elbo
        
        