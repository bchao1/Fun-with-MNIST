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
from VAE import VAE
import torchvision

if __name__ == '__main__':
    
    epochs = 50
    batch_size = 100
    latent_dim = 2
    dataloader = utils.get_dataloader(batch_size)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    net = VAE(latent_dim = latent_dim).to(device)
    optim = utils.get_optim(net, 0.0002)
    
    loss_log = []
    
    criterion = nn.MSELoss()
    
    result = None
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, _) in enumerate(dataloader):
            real_img = real_img.to(device)
            
            if result is None:
                result = real_img
                
            reconstructed = net(real_img)
            
            reconstruction_loss = criterion(reconstructed, real_img)
            kl_loss = utils.kl_loss(net.z_mean, net.z_sigma)
            
            loss = kl_loss + reconstruction_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_log.append(loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, loss_log, kl_loss)
            
        if epoch_i == 1:
            torchvision.utils.save_image(result, 
                                         os.path.join(sample_dir, 'orig.png'), 
                                         nrow = 10)
        reconstructed = net(result)
        utils.save_image(reconstructed, 10, epoch_i, step_i + 1, sample_dir)
                
        utils.save_model(net, optim, loss_log, checkpoint_dir, 'autoencoder.ckpt')
        
            
    
    
    
    
    