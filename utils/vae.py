# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:08:28 2018

@author: USER
"""

def kl_loss(z_mean, z_std):
    z_mean_sqr = z_mean**2
    z_std_sqr = z_std**2
    return 0.5 * (z_mean_sqr + z_std_sqr - torch.log(z_std_sqr) - 1).mean()
