# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""

import torch
import torch.nn as nn
import numpy as np
import utils
from Autoencoder import Autoencoder

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
    
    net = Autoencoder().to(device)
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
            loss = criterion(reconstructed, real_img)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_log.append(loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, loss_log)
            if (step_i + 1) % 200 == 0:
                reconstructed = net(result)
                utils.save_image(reconstructed, 10, epoch_i, step_i + 1, sample_dir)
                
        utils.save_model(net, optim, loss_log, checkpoint_dir, 'G.ckpt')
        
            
    
    
    
    
    