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
import utils.adversarial_ae as ae_utils
from adverse_AE import Adversarial_AE, Discriminator
import torchvision
import torch.optim as optim

if __name__ == '__main__':
    
    epochs = 50
    batch_size = 100
    latent_dim = 2
    reg = True
    dataloader = utils.get_dataloader(batch_size, pad = False)
    device = utils.get_device()
    step_per_epoch = np.ceil(dataloader.dataset.__len__() / batch_size)
    sample_dir = './samples'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(sample_dir, checkpoint_dir)
    
    AE = Adversarial_AE(latent_dim = latent_dim).to(device)
    D = Discriminator(latent_dim = latent_dim).to(device)
    
    ae_optim = optim.Adam(AE.parameters())
    d_optim = optim.Adam(D.parameters())
    
    rec_log = []
    d_log = []
    
    rec_criterion = nn.MSELoss().to(device)
    discrim_criterion = nn.BCELoss().to(device)
    
    result = None
    for epoch_i in range(1, epochs + 1):
        for step_i, (img, _) in enumerate(dataloader):
            N = img.shape[0]
            real_label = torch.ones(N).to(device)
            fake_label = torch.zeros(N).to(device)
            
            img = img.view(N, -1).to(device)
            if result is None:
                result = img
            
            # Reconstruction phase
            reconstructed = AE(img)
            
            loss = rec_criterion(reconstructed, img)
            
            ae_optim.zero_grad()
            loss.backward()
            ae_optim.step()
            rec_log.append(loss.item())
            
            # Discriminator phase
            z = torch.randn(N, latent_dim).to(device)
            code = AE.encoder(img)
            fake_score = D(code)
            real_score = D(z)
            
            real_loss = discrim_criterion(real_score, real_label)
            fake_loss = discrim_criterion(fake_score, fake_label)
            loss = real_loss + fake_loss
            
            d_optim.zero_grad()
            loss.backward()
            d_optim.step()
            d_log.append(loss.item())
            
            code = AE.encoder(img)
            fake_score = D(code)
            loss = discrim_criterion(fake_score, real_label)
            
            ae_optim.zero_grad()
            loss.backward()
            ae_optim.step()
            
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, rec_log, d_log)
            
        if epoch_i == 1:
            torchvision.utils.save_image(result.reshape(-1, 1, 28, 28), 
                                         os.path.join(sample_dir, 'orig.png'), 
                                         nrow = 10)
        reconstructed = AE(result)
        utils.save_image(reconstructed.reshape(-1, 1, 28, 28), 10, epoch_i, 
                         step_i + 1, sample_dir)
                
        utils.save_model(AE, ae_optim, rec_log, checkpoint_dir, 'AE.ckpt')
        utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')

    ae_utils.plot_manifold(AE.encoder, device, dataloader.dataset, 
                        dataloader.dataset.__len__(), sample_dir)
        
            
    
    
    
    
    