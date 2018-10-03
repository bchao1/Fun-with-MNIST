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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_code = ['#D9E500', '#E0A200', '#DC5600', '#D80C00', '#89E000', 
              '#CF007D', '#CB00BE', '#9200C7', '#4E00C3', '#0E00BF']

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

def show_process(epoch, step, step_per_epoch, loss_log):
    print('Epoch [{}], Step [{}/{}], Loss [{:8f}]'.format(
            epoch, step, step_per_epoch, loss_log[-1]))
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


def random_generation(model, device, latent_dim, sample_dir):
    code = torch.Tensor(100, latent_dim).uniform_().to(device)
    output = model(code)
    filename = os.path.join(sample_dir, 'random.png')
    torchvision.utils.save_image(output, filename, 10)
    return

def manifold_walk(model, steps, sample_dir):
    results = []
    for i in np.linspace(0, 1, steps + 1):
        for j in np.linspace(0, 1, steps + 1):
            code = torch.zeros(1, 2, 1, 1)
            code[:, 0, :, :] = i
            code[:, 1, :, :] = j
            results.append(code)
    results = torch.cat(results)
    output = model(results)
    filename = os.path.join(sample_dir, 'manifold.png')
    torchvision.utils.save_image(output, filename, steps + 1)
    return

def L2_reg(code):
    return code.view(-1).norm(p = 2).mean()

def plot_manifold(encoder, dataset, num_points, sample_dir):
    plt.figure(figsize=(6, 6), dpi=100)
    labels = []
    data = []
    for i in range(num_points):
        img, label = dataset.__getitem__(i)
        labels.append(label)
        data.append(img.unsqueeze(0))
    data = torch.cat(data, 0)
    code = encoder(data).squeeze()
    x, y = code.detach().numpy().transpose()
    for i in range(num_points):
        plt.scatter(x[i], y[i], marker = '.', c = color_code[labels[i]])
    
    patches = []
    for i in range(len(color_code)):
        patches.append(mpatches.Patch(color=color_code[i], label=i))
    plt.legend(handles = patches)
    plt.savefig(os.path.join(sample_dir, 'manifold_scatter.png'))
    