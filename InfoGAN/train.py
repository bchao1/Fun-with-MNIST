# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""

import torch
import torch.nn as nn
import numpy as np
import utils
from InfoGAN import Generator, Discriminator

if __name__ == '__main__':
    
    epochs = 200
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
    
    criterion = nn.BCELoss()
    cont_criterion = nn.MSELoss()
    
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, _) in enumerate(dataloader):
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            tags = utils.get_random_tags(batch_size, 2, 10)
            
            target_cont_tag = tags[:, 2]
            target_discrete_tag = tags[2: 13]
            # Train D
            
            real_img = real_img.to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z, tags)
            
            real_score, _, _ = D(real_img)
            fake_score, fake_cont_tag, fake_discrete_tag = D(fake_img)
            
            real_loss = criterion(real_score, real_labels)
            fake_loss = criterion(fake_score, fake_labels)
            discrim_loss = real_loss + fake_loss
            
            discrete_loss = criterion(fake_discrete_tag, target_discrete_tag)
            cont_loss = cont_criterion(fake_cont_tag, target_cont_tag)
            info_loss = cont_loss + discrete_loss
            
            d_loss = discrim_loss + info_loss
            
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            d_log.append(d_loss.item())
            
            # Train G
            
            z = torch.randn(batch_size, latent_dim).to(device)
            tags = utils.get_random_tags(batch_size, 2, 10)
            
            target_cont_tag = tags[:, 2]
            target_discrete_tag = tags[2: 13]
            
            fake_img = G(z, tags)
            fake_score, fake_cont_tag, fake_discrete_tag = D(fake_img)
            
            discrim_loss = criterion(fake_score, real_labels)
            
            discrete_loss = criterion(fake_discrete_tag, target_discrete_tag)
            cont_loss = cont_criterion(fake_cont_tag, target_cont_tag)
            info_loss = discrete_loss + cont_loss
            
            g_loss = discrim_loss + info_loss
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_log.append(g_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, g_log, d_log)
            if (step_i + 1) % 200 == 0:
                utils.save_image(fake_img, 10, epoch_i, step_i + 1, sample_dir)
                
        utils.save_model(G, g_optim, g_log, checkpoint_dir, 'G.ckpt')
        utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')
        
            
    
    
    
    
    