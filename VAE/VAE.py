# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:21:27 2018

@author: USER
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

class Autoencoder(nn.Module):
    def __init__(self, latent_dim = 100):
        super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 1024, 2, 1),
                nn.LeakyReLU(0.2)
                )
        self.enc_log_sigma = nn.Linear(1024, self.latent_dim)
        self.enc_mu = nn.Linear(1024, self.latent_dim)
        
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.latent_dim, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1, bias = False),
                nn.Sigmoid()
                )
        
    def sample_latent(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        
        self.z_mean = mu
        self.z_sigma = sigma
        
        std_z = torch.Tensor(sigma.shape).normal_()
        return mu + sigma * Variable(std_z, False)
        
    def forward(self, input):
        code = self.encoder(input)
        flatten = code.view(-1, 1024)
        
        sigma = self.enc_log_sigma(flatten)
        mu = self.enc_mu(flatten)
        
        z = self.sample_latent(mu, sigma).unsqueeze_(2).unsqueeze_(3)
        return self.decoder(z)

    
if __name__ == '__main__':
    AE = Autoencoder(2)
    x = torch.randn(100, 1, 32, 32)
    o = AE(x)