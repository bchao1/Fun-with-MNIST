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
from torch.autograd import Variable, grad

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
    return optim.Adam(model.parameters(), betas = [0.5, 0.9], lr = lr)

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

def interpolate(real_img, fake_img):
    theta = np.random.uniform()
    sample = theta * real_img + (1 - theta) * fake_img
    return sample

def gradient_norm(model, real_img, fake_img):
    _input = interpolate(real_img, fake_img)
    _input = Variable(_input, requires_grad = True)
    score = model(_input)
    gradient = grad(outputs = score, 
                    inputs = _input, 
                    create_graph = True,
                    retain_graph = True,
                    only_inputs = True)[0]
    grad_norm = ((gradient.view(-1).norm(p = 2) - 1)**2).mean()
    return grad_norm

    
    
    
    