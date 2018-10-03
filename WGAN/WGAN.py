# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:21:27 2018

@author: USER
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim = 100):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
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
        
    def forward(self, input):
        output = input.unsqueeze(2).unsqueeze(3)
        return self.net(output)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
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
                nn.Conv2d(512, 1, 2, 1)
                )
    def forward(self, input):
        return self.net(input).view(-1)
    
if __name__ == '__main__':
    z = torch.randn(5, 100)
    g = Generator(100)
    d = Discriminator()
    o = g(z)
    s = d(o)
    