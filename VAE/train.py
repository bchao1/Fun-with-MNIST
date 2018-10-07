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
    
    epochs = 100
    batch_size = 100
    latent_dim = 2
    dataloader = utils.get_dataloader(batch_size)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    net = VAE(latent_dim = latent_dim).to(device)
    optim = torch.optim.Adam(net.parameters())
    
    rec_log = []
    kl_log = []
    
    criterion = nn.BCELoss(reduction = 'sum')
    
    result = None
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, _) in enumerate(dataloader):
            N = real_img.shape[0]
            real_img = real_img.view(N, -1).to(device)
            
            if result is None:
                result = real_img
                
            reconstructed, mu, logvar = net(real_img)
            
            reconstruction_loss = criterion(reconstructed, real_img)
            kl_loss = utils.kl_loss(mu, logvar)
            
            loss = kl_loss + reconstruction_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            rec_log.append(reconstruction_loss.item())
            kl_log.append(kl_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, rec_log, kl_log)
            
        if epoch_i == 1:
            torchvision.utils.save_image(result.reshape(-1, 1, 28, 28), 
                                         os.path.join(sample_dir, 'orig.png'), 
                                         nrow = 10)
        reconstructed, _, _ = net(result)
        utils.save_image(reconstructed.reshape(-1, 1, 28, 28), 10, epoch_i, step_i + 1, sample_dir)
        sample = net.decoder(torch.randn((100, 2)).to(device))
        torchvision.utils.save_image(sample.reshape(-1, 1, 28, 28), 
                                         os.path.join(sample_dir, 'sample_{}.png'.format(epoch_i)), 
                                         nrow = 10)
                
        utils.save_model(net, optim, rec_log, checkpoint_dir, 'autoencoder.ckpt')

    steps = 50
    z = utils.box_muller(steps).to(device)
    result = net.decoder(z)
    torchvision.utils.save_image(result.reshape(-1, 1, 28, 28), 
                                 os.path.join(sample_dir, 'manifold.png'), 
                                 nrow = steps)
        
            
    
    
    
    
    