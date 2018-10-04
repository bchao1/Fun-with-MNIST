# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:34:29 2018

@author: USER
"""

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

color_code = ['#D9E500', '#E0A200', '#DC5600', '#D80C00', '#89E000', 
              '#CF007D', '#CB00BE', '#9200C7', '#4E00C3', '#0E00BF']

def show_process(epoch, step, step_per_epoch, loss_log):
    print('Epoch [{}], Step [{}/{}], Loss [{:8f}]'.format(
            epoch, step, step_per_epoch, loss_log[-1]))
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
    