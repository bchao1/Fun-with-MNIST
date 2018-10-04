# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:21:27 2018

@author: USER
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim = 100, class_dim = 10):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        
        self.net = nn.Sequential(
                nn.ConvTranspose2d(self.latent_dim + self.class_dim, 512, 4, 2, 1, bias = False),
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
        
    def forward(self, _input, _class):
        concat = torch.cat((_input, _class), 1)
        concat = concat.unsqueeze(2).unsqueeze(3)
        output = self.net(concat)
        return output

class Discriminator(nn.Module):
    def __init__(self, class_dim = 10):
        super(Discriminator, self).__init__()
        
        self.class_dim = class_dim
        self.upsample = nn.Sequential(
                nn.ConvTranspose2d(self.class_dim, 128, 2, 1, 0, bias = False),
                nn.LeakyReLU(0.2)
                )
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
                nn.LeakyReLU(0.2)
                )
        self.output_layer = nn.Sequential(
                nn.Conv2d(512 + 128, 1, 2, 1),
                nn.Sigmoid()
                )

    def forward(self, _input, _class):
        class_map = _class.unsqueeze(2).unsqueeze(3)
        class_feature = self.upsample(class_map)
        noise_feature = self.net(_input)
        concat = torch.cat((noise_feature, class_feature), 1)
        return self.output_layer(concat).view(-1)
    
if __name__ == '__main__':
    c = torch.randn(5, 10)
    z = torch.randn(5, 100)
    g = Generator()
    d = Discriminator()
    o = g(z, c)
    s = d(o, c)
    