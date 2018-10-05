# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""

import os
import torch
import torch.nn as nn
import numpy as np
import utils
from dense_CGAN import Generator, Discriminator
import torchvision

if __name__ == '__main__':
    
    epochs = 200
    batch_size = 100
    latent_dim = 100
    dataloader = utils.get_dataloader(batch_size, pad = False)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    G = Generator(latent_dim = latent_dim).to(device)
    D = Discriminator().to(device)
    
    g_optim = utils.get_optim(G, 0.0005)
    d_optim = utils.get_optim(D, 0.0005)
    
    g_log = []
    d_log = []
    
    criterion = nn.BCELoss()
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, class_label) in enumerate(dataloader):
            N = real_img.shape[0]
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            onehot_class = utils.to_onehot(class_label, 10).to(device)
            
            mismatch_label = utils.get_label_mismatch(class_label, 10)
            mismatch_class = utils.to_onehot(mismatch_label, 10)
            # Train D
            
            real_img = real_img.view(N, -1).to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z, onehot_class)
            
            real_img_right_class = D(real_img, onehot_class)
            real_img_wrong_class = D(real_img, mismatch_class)
            fake_img_right_class = D(fake_img, onehot_class)
            
            right_loss = criterion(real_img_right_class, real_labels)
            wrong_loss = (criterion(fake_img_right_class, fake_labels) + 
                         criterion(real_img_wrong_class, fake_labels)) * 0.5
            
            d_loss = right_loss + wrong_loss
            
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            d_log.append(d_loss.item())
            
            # Train G
            
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z, onehot_class)
            
            fake_score = D(fake_img, onehot_class)
            
            g_loss = criterion(fake_score, real_labels)
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_log.append(g_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, g_log, d_log)
        
        if epoch_i == 1:
            torchvision.utils.save_image(real_img.reshape(-1, 1, 32, 32), 
                                         os.path.join(sample_dir, 'real.png'),
                                         nrow = 10)
                
        utils.save_model(G, g_optim, g_log, checkpoint_dir, 'G.ckpt')
        utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')
        utils.generate_classes(G, latent_dim, device, 10, epoch_i, sample_dir)
        
            
    
    
    
    
    