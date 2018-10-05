# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:35:00 2018

@author: USER
"""

import torch
import torch.nn as nn


class Adversarial_AE(nn.Module):
    def __init__(self, latent_dim = 2):
        super(Adversarial_AE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
                nn.Linear(784, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, self.latent_dim)
                )
        
        self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 784),
                nn.Sigmoid()
                )
        
    def forward(self, _input):
        return self.decoder(self.encoder(_input))
        
class Discriminator(nn.Module):
    def __init__(self, latent_dim = 2):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
                nn.Linear(self.latent_dim, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1),
                nn.Sigmoid()
                )
    def forward(self, _input):
        return self.net(_input).view(-1)