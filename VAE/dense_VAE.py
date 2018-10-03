# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:15:31 2018

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:21:27 2018

@author: USER
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class dense_VAE(nn.Module):
    def __init__(self, latent_dim = 100):
        super(dense_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(1024, 2048),
                                     nn.BatchNorm1d(2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     )
                                     
                                     
        self.enc_log_sigma = nn.Linear(128, self.latent_dim)
        self.enc_mu = nn.Linear(128, self.latent_dim)
        
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Linear(128, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     nn.Linear(512, 2048),
                                     nn.BatchNorm1d(2048),
                                     nn.ReLU(),
                                     nn.Linear(2048, 1024),
                                     nn.Sigmoid()
                                     )
        
    def sample_latent(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        
        self.z_mean = mu
        self.z_sigma = sigma
        
        std_z = torch.Tensor(sigma.shape).normal_()
        if torch.cuda.is_available():
            std_z = std_z.cuda()
        
        return mu + sigma * Variable(std_z, False)
        
    def forward(self, input):
        code = self.encoder(input)
        
        sigma = self.enc_log_sigma(code)
        mu = self.enc_mu(code)
        
        z = self.sample_latent(mu, sigma)
        return self.decoder(z)

    
if __name__ == '__main__':
    AE = dense_VAE(2)
    x = torch.randn(100, 1024)
    o = AE(x)