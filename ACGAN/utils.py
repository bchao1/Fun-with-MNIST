# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:34:29 2018

@author: USER
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision
import numpy as np

def get_dataloader(batch_size):
    transform = transforms.Compose([
            transforms.Pad(padding = 2, padding_mode = 'edge'),
            transforms.ToTensor()
            ])
    
    dataset = MNIST(root = '../data', train = True, download = False, 
                    transform = transform)
    dataloader = DataLoader(dataset = dataset, batch_size = batch_size, 
                            shuffle = True)
    return dataloader

def makedirs(sample_dir, checkpoint_dir):
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def get_optim(model, lr):
    return optim.Adam(model.parameters(), betas = [0.5, 0.999], lr = lr)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def show_process(epoch, step, step_per_epoch, g_log, d_log, classifier_log):
    print('Epoch [{}], Step [{}/{}], Losses: G [{:8f}], D [{:8f}], Classifier[{:8f}]'.format(
            epoch, step, step_per_epoch, g_log[-1], d_log[-1], classifier_log[-1]))
    return    

def save_model(model, optim, logs, ckpt_dir, filename):
    file_path = os.path.join(ckpt_dir, filename)
    state = {'model': model.state_dict(),
             'optim': optim.state_dict(),
             'logs': tuple(logs),
             'steps': len(logs)}
    torch.save(state, file_path)
    return

def save_image(img, nrow, epoch, step, sample_dir):
    filename = 'epoch_{}_step_{}.png'.format(epoch, step)
    file_path = os.path.join(sample_dir, filename)
    torchvision.utils.save_image(img, file_path, nrow)
    return

def to_onehot(labels, classes):
    batch_size = labels.shape[0]
    onehot = torch.zeros(batch_size, classes)
    for i in range(batch_size):
        onehot[i][labels[i]] = 1
    return onehot

def generate_classes(model, latent_dim, device, num_classes, epoch, sample_dir):
    class_label = np.random.randint(num_classes)
    class_vec = torch.zeros(num_classes).to(device)
    class_vec[class_label] = 1
    class_vec.unsqueeze_(0)
    
    images = []
    for i in range(100):
        z = torch.randn(latent_dim).unsqueeze(0).to(device)
        output = model(z, class_vec)
        images.append(output)
    images = torch.cat(images, 0)
    filename = 'epoch_{}_class_{}.png'.format(epoch, class_label)
    torchvision.utils.save_image(images, os.path.join(sample_dir, filename), 
                                 nrow = 10)
    return
        
    
    
    
    