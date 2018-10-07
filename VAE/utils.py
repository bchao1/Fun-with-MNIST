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


def get_dataloader(batch_size, pad = False):
    if pad:
        transform = transforms.Compose([
                transforms.Pad(padding = 2, padding_mode = 'edge'),
                transforms.ToTensor()
                ])
    else:
        transform = transforms.ToTensor()
    
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
    return optim.Adam(model.parameters(), betas = [0.9, 0.999], lr = lr)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def show_process(epoch, step, step_per_epoch, rec, kl):
    print('Epoch [{}], Step [{}/{}], Rec Loss [{:8f}], KL Loss [{:8f}]'.format(
            epoch, step, step_per_epoch, rec[-1], kl[-1]))
    return    

def save_model(model, optim, logs, ckpt_dir, filename):
    file_path = os.path.join(ckpt_dir, filename)
    state = {'model': model.state_dict(),
             'optim': optim.state_dict(),
             'logs': tuple(logs),
             'steps': len(logs)}
    torch.save(state, file_path)
    return

def load_model(model, filepath):
    state = torch.load(filepath)
    model.load_state_dict(state['model'])

def save_image(img, nrow, epoch, step, sample_dir):
    filename = 'epoch_{}.png'.format(epoch, step)
    file_path = os.path.join(sample_dir, filename)
    torchvision.utils.save_image(img, file_path, nrow)
    return


def random_generation(model, device, latent_dim, sample_dir):
    code = torch.Tensor(100, latent_dim).uniform_().to(device)
    output = model(code)
    filename = os.path.join(sample_dir, 'random.png')
    torchvision.utils.save_image(output, filename, 10)
    return

def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def box_muller(steps):
    p = []
    u1 = np.linspace(0.01, 0.99, steps)
    u2 = np.linspace(0.01, 0.99, steps)
    for i in range(len(u1)):
        for j in range(len(u2)):
            z1 = np.sqrt(-2*np.log(u1[i]))*np.cos(2*np.pi*u2[j])
            z2 = np.sqrt(-2*np.log(u1[i]))*np.sin(2*np.pi*u2[j])
            coor = [z1, z2]
            p.append(torch.tensor(coor, dtype = torch.float).unsqueeze(0))
    return torch.cat(p, 0)
    