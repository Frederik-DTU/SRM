# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:12:32 2021

@author: Frederik
"""

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from typing import List, Any

import pandas as pd 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Sources:
#https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
#http://adamlineberry.ai/vae-series/vae-code-experiments
#https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa

#%% VAE

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
        eps = torch.FloatTensor(std.size()).normal_()
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
        z = self.reparameterize(mu, var)
        
        return z

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1**2-x2**2

#%% Data loader

class sim_plot_3d_fun(object):
    def __init__(self,
                 x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = 50000,
                 name_path = 'para_data.csv',
                 seed = 100,
                 fig_size = (8,6)):
        
        self.x1_fun = x1_fun
        self.x2_fun = x2_fun
        self.x3_fun = x3_fun
        self.N_sim = N_sim
        self.name_path = name_path
        self.seed = seed
        self.fig_size = fig_size
        
    def sim_3d(self):
    
        np.random.seed(self.seed)
        x1 = self.x1_fun(self.N_sim)
        x2 = self.x2_fun(self.N_sim)
        
        x3 = self.x3_fun(x1, x2)
        
        df = np.vstack((x1, x2, x3))
        
        pd.DataFrame(df).to_csv(self.name_path)
        
        return
    
    def plot_geodesic_in_Z_2d(self, *args):
        
        fig = plt.figure(figsize=self.fig_size)
        
        for arg in args:
            lab = arg[1]
            x = arg[0][:,0]
            y = arg[0][:,1]
            plt.plot(x.detach().numpy(), y.detach().numpy(), '-*', label=lab)
            
        plt.xlabel('t')
        plt.ylabel('')
        plt.grid()
        plt.show()
        plt.legend()
        plt.tight_layout()
        plt.title('Geodesic in Z')
        
        return
        
    
    def true_plot_3d(self, x1_grid, x2_grid, N_grid, title):
        
        x1_grid = np.linspace(x1_grid[0], x1_grid[1], num = N_grid)
        x2_grid = np.linspace(x2_grid[0], x2_grid[1], num = N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X3 = self.x3_fun(X1, X2)
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        ax.plot_surface(X1, X2, X3, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
        ax.set_title(title);
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.tight_layout()
        
        plt.show()
        
    def plot_data_surface_3d(self, x1, x2, x3, title="Test"):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        ax.plot_trisurf(x1, x2, x3,
                cmap='viridis', edgecolor='none');
        ax.set_title(title);
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.tight_layout()
        
        plt.show()
        
    def plot_data_scatter_3d(self, x1, x2, x3, N_grid, title="Test"):
        
        fig = plt.figure(figsize=self.fig_size)
        ax = plt.axes(projection="3d")
        
        x1_grid = np.linspace(min(x1), max(x1), num = N_grid)
        x2_grid = np.linspace(min(x2), max(x2), num = N_grid)
        
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        X3 = self.x3_fun(X1, X2)
        ax.plot_surface(
        X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        
        ax.legend()
        
        p = ax.scatter3D(x1, x2, x3, color='black')
        
        plt.tight_layout()
        
        plt.show()
        
    def read_data(self):
        
        df = pd.read_csv(self.name_path, index_col=0)

        dat = torch.Tensor(df.values)
        
        return dat

#%% Training

file_path_name = 'para_data.csv'
data_obj = sim_plot_3d_fun(N_sim=50000, name_path = file_path_name)
generate_data = False
try_cuda = True
plot_results = False

if generate_data:
    data_obj.sim_3d()

if try_cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'
    
dat = data_obj.read_data()
dat = torch.transpose(dat, 0, 1)

epochs = 10#100000
batch_size = 100
lr = 0.0001

trainloader = DataLoader( dataset = dat , batch_size= batch_size , shuffle = True)
reconstruction_function = nn.MSELoss(reduction='sum')

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

model = VAE_3d_surface().to(device)

def loss_fun(x_rec, x, mu, var):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    logvar = torch.log(var)
    BCE = reconstruction_function(x_rec, x) 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KL divergence
    return BCE + KLD

optimizer = optim.SGD(model.parameters(), lr=lr)
#criterion = nn.BCELoss(reduction='sum')

train_loss = []
val_loss = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x in trainloader:
        #x = x
        #x = x.view(x.size(0), -1)
        #x.requires_grad=True
        x = x.to(device)
        _, x_rec, mu, var = model(x)
        #bce_loss = criterion(x_rec, data, mu, logvar)
        loss = loss_fun(x_rec, x, mu, var)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        running_loss += loss.item()
        optimizer.step()
        
    train_epoch_loss = running_loss/len(trainloader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
    train_loss.append(train_epoch_loss)

torch.save(model.state_dict(), 'para_3d')
