# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""
import sys
sys.path.append('..')
import os
import torch
import torch.nn as nn
import numpy as np
import utils.general as utils
from LSGAN import Generator, Discriminator
import torchvision

if __name__ == '__main__':
    
    epochs = 100
    batch_size = 100
    latent_dim = 100
    dataloader = utils.get_dataloader(batch_size)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    G = Generator(latent_dim = latent_dim).to(device)
    D = Discriminator().to(device)
    
    g_optim = utils.get_optim(G, 0.0002)
    d_optim = utils.get_optim(D, 0.0002)
    
    g_log = []
    d_log = []
    
    fix_z = torch.randn(batch_size, latent_dim).to(device)
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, _) in enumerate(dataloader):
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train D
            
            real_img = real_img.to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z)
            
            real_score = D(real_img)
            fake_score = D(fake_img)
            
            real_loss = ((real_score - 1)**2).mean()
            fake_loss = (fake_score**2).mean()
            
            d_loss = (real_loss + fake_loss) * 0.5
            
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            d_log.append(d_loss.item())
            
            # Train G
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z)
            
            fake_score = D(fake_img)
            
            g_loss = ((fake_score - 1)**2).mean() * 0.5
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_log.append(g_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, g_log, d_log)
        
        if epoch_i == 1:
            torchvision.utils.save_image(real_img, 
                                         os.path.join(sample_dir, 'real.png'),
                                         nrow = 10)
        fake_img = G(fix_z)
        utils.save_image(fake_img, 10, epoch_i, step_i + 1, sample_dir)
                
        utils.save_model(G, g_optim, g_log, checkpoint_dir, 'G.ckpt')
        utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')
        
            
    
    
    
    
    