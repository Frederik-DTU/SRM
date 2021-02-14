# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:52:13 2021

@author: Frederik
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:34:11 2021

@author: Frederik
"""
import torch
from torch import nn
from torch.autograd import Variable
from typing import List, Any


#Sources:
#https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
#https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb

class VAE_3d_surface(nn.Module):
    def __init__(self,
                 fc_h: List[int] = [3,100],
                 fc_g: List[int] = [2, 100, 3],
                 fc_mu: List[int] = [100, 2],
                 fc_var: List[int] = [100, 2],
                 fc_h_act: List[Any] = [nn.ELU],
                 fc_g_act: List[Any] = [nn.ELU, nn.Identity],
                 fc_mu_act: List[Any] = [nn.Identity],
                 fc_var_act: List[Any] = [nn.Sigmoid],
                 device: str = 'cuda'
                 ):
        super(VAE_3d_surface, self).__init__()
    
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
        self.device = device
        
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
        
    def reparameterize(self, mu, var):
        
        std = torch.sqrt(var)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
        
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
        z = self.reparameterize(mu, var)
        x_rec = self.g(z)
        
        return z, x_rec, mu, var
    
    def get_decoded(self, z):
        
        x_rec = self.g(z)
        
        return x_rec
    
    def get_encoded(self, x):
        
        x = self.h(x)
        mu = self.mu_net(x)
        var = self.var_net(x)
        #z = self.reparameterize(mu, var)
        
        return mu
        
#Simple test
#vae = para_VAE()
#x_test = torch.tensor([1,2,3], dtype=torch.torch.float)
#x_rec = vae.forward(x_test)


        
        