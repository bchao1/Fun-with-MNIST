# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 01:21:50 2018

@author: USER
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim = 100, class_dim = 10):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        
        self.class_embedding = nn.Sequential(
                nn.Linear(self.class_dim, 200),
                nn.ReLU()
                )
        
        self.feature_embedding = nn.Sequential(
                nn.Linear(self.latent_dim, 1000),
                nn.ReLU()
                )
        self.net = nn.Sequential(
                nn.Linear(1200, 1000),
                nn.ReLU(),
                nn.Linear(1000, 784),
                nn.Sigmoid()
                )
        
    def forward(self, _input, _class):
        x1 = self.feature_embedding(_input)
        x2 = self.class_embedding(_class)
        concat = torch.cat((x1, x2), 1)
        return self.net(concat)

class Discriminator(nn.Module):
    def __init__(self, class_dim = 10):
        super(Discriminator, self).__init__()
        
        self.class_dim = class_dim
        
        self.class_embedding = nn.Sequential(
                nn.Linear(self.class_dim, 200),
                nn.ReLU()
                )
        
        self.feature_embedding = nn.Sequential(
                nn.Linear(784, 800),
                nn.ReLU()
                )
        
        self.net = nn.Sequential(
                nn.Linear(1000, 1000),
                nn.ReLU(),
                nn.Linear(1000, 1),
                nn.Sigmoid()
                )

    def forward(self, _input, _class):
        x1 = self.feature_embedding(_input)
        x2 = self.class_embedding(_class)
        concat = torch.cat((x1, x2), 1)
        return self.net(concat).view(-1)


if __name__ == '__main__':
    z = torch.randn(5, 100)
    c = torch.randn(5, 10)
    g = Generator()
    d = Discriminator()
    o = g(z, c)
    s = d(o, c)