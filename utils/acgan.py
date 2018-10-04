# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:59:37 2018

@author: USER
"""
import os
import torch
import torchvision

def generate_classes(model, latent_dim, device, num_classes, epoch, sample_dir):  
    
    images = []
    for i in range(num_classes):
        class_vec = torch.zeros(num_classes).to(device)
        class_vec[i] = 1
        class_vec.unsqueeze_(0)
        for _ in range(10):
            z = torch.randn(latent_dim).unsqueeze(0).to(device)
            output = model(z, class_vec)
            images.append(output)
            
    images = torch.cat(images, 0)
    filename = 'epoch_{}.png'.format(epoch)
    torchvision.utils.save_image(images, os.path.join(sample_dir, filename), 
                                 nrow = 10)
    return

def fix_noise(model, latent_dim, device, num_classes, sample_dir):
    images = []
    for _ in range(10):
        fix_z = torch.randn(latent_dim).unsqueeze(0).to(device)
        for i in range(num_classes):
            class_vec = torch.zeros(num_classes).to(device)
            class_vec[i] = 1
            class_vec.unsqueeze_(0)
            output = model(fix_z, class_vec)
            images.append(output)
    images = torch.cat(images, 0)
    filename =  os.path.join(sample_dir, 'fix_noise.png')
    torchvision.utils.save_image(images, filename, nrow = num_classes)
    return
    
def show_process(epoch, step, step_per_epoch, g_log, d_log, classifier_log):
    print('Epoch [{}], Step [{}/{}], Losses: G [{:8f}], D [{:8f}], Classifier[{:8f}]'.format(
            epoch, step, step_per_epoch, g_log[-1], d_log[-1], classifier_log[-1]))
    return 