# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""

import torch
import torch.nn as nn
import numpy as np
import utils
from GAN import Generator, Discriminator

if __name__ == '__main__':
    
    epochs = 200
    batch_size = 64
    latent_dim = 100
    d_updates = 5
    gp_weight = 10
    
    dataloader = utils.get_dataloader(batch_size)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    G = Generator(latent_dim = latent_dim).to(device)
    D = Discriminator().to(device)
    
    g_optim = utils.get_optim(G, 0.0001)
    d_optim = utils.get_optim(D, 0.0001)
    
    g_log = []
    d_log = []
    
    criterion = nn.BCELoss()
    
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, _) in enumerate(dataloader):
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            # Train D
            
            d_loss_avg = 0
            for _ in range(d_updates):
                real_img = real_img.to(device)
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_img = G(z)
                
                real_score = D(real_img)
                fake_score = D(fake_img)
                grad_penalty = utils.gradient_norm(D, real_img, fake_img)
                d_loss = fake_score - real_score + grad_penalty * gp_weight
                d_loss_avg += d_loss
                
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
            d_loss_avg /= d_updates
            d_log.append(d_loss_avg.item())
            
            # Train G
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z)
            
            fake_score = D(fake_img)
            
            g_loss = -fake_score
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_log.append(g_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, g_log, d_log)
            if (step_i + 1) % 200 == 0:
                utils.save_image(fake_img, 10, epoch_i, step_i + 1, sample_dir)
                
        utils.save_model(G, g_optim, g_log, checkpoint_dir, 'G.ckpt')
        utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')
        
            
    
    
    
    
    