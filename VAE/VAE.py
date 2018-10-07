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

class VAE(nn.Module):
    def __init__(self, latent_dim = 100):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(784, 400),
                                     nn.BatchNorm1d(400),
                                     nn.ReLU()
                                     )
                                     
                                     
        self.enc_log_sigma = nn.Linear(400, self.latent_dim)
        self.enc_mu = nn.Linear(400, self.latent_dim)
        
        self.decoder = nn.Sequential(nn.Linear(self.latent_dim, 400),
                                     nn.BatchNorm1d(400),
                                     nn.ReLU(),
                                     nn.Linear(400,784),
                                     nn.Sigmoid()
                                     )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_mu(h1), self.enc_log_sigma(h1)
        
    def sample_latent(self, mu, log_sigma):
        sigma = torch.exp(0.5*log_sigma)
        
        eps = torch.Tensor(sigma.shape).normal_()
        if torch.cuda.is_available():
            eps = eps.cuda()
        
        return eps.mul(sigma).add_(mu)
        
    def forward(self, input):
        mu, logvar = self.encode(input)
        
        z = self.sample_latent(mu, logvar)
        return self.decoder(z), mu, logvar
