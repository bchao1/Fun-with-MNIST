# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:51:41 2018

@author: USER
"""
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np
import utils.general as utils
import utils.acgan as acgan_utils
from ACGAN import Generator, Discriminator

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
    
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    g_optim = utils.get_optim(G, 0.0002)
    d_optim = utils.get_optim(D, 0.0002)
    
    g_log = []
    d_log = []
    classifier_log = []
    
    discrim_criterion = nn.BCELoss()
    aux_criterion = nn.CrossEntropyLoss()

    for epoch_i in range(1, epochs + 1):
        for step_i, (real_img, class_label) in enumerate(dataloader):
            
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            
            onehot_class = utils.to_onehot(class_label, 10).to(device)
            # Train D
            
            real_img = real_img.to(device)
            class_label = class_label.to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_img = G(z, onehot_class)
            
            real_score, real_aux = D(real_img)
            fake_score, fake_aux = D(fake_img)
            
            real_discrim_loss = discrim_criterion(real_score, real_labels)
            fake_discrim_loss = discrim_criterion(fake_score, fake_labels)
            
            real_aux_loss = aux_criterion(real_aux, class_label)
            fake_aux_loss = aux_criterion(fake_aux, class_label)
            
            discrim_loss = real_discrim_loss + fake_discrim_loss
            aux_loss = real_aux_loss + fake_aux_loss
            d_total_loss = aux_loss + discrim_loss
            
            d_optim.zero_grad()
            d_total_loss.backward()
            d_optim.step()
            d_log.append(d_total_loss.item())
            classifier_log.append(aux_loss.item())
            
            # Train G
            
            z = torch.randn(batch_size, latent_dim).to(device)
            
            fake_img = G(z, onehot_class)
            
            fake_score, fake_aux = D(fake_img)
            
            fake_discrim_loss = discrim_criterion(fake_score, real_labels)
            fake_aux_loss = aux_criterion(fake_aux, class_label)
            g_loss = fake_discrim_loss + fake_aux_loss
            
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_log.append(g_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, step_per_epoch, 
                               g_log, d_log, classifier_log)
            
        if epoch_i % 5 == 0:
            utils.save_model(G, g_optim, g_log, checkpoint_dir, 'G.ckpt')
            utils.save_model(D, d_optim, d_log, checkpoint_dir, 'D.ckpt')
            acgan_utils.generate_classes(G, latent_dim , device, 10, epoch_i, sample_dir)
        
    acgan_utils.fix_noise(G, latent_dim, device, 10, sample_dir)
    
    
    