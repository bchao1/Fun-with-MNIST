# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:56:54 2018

@author: USER
"""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

color_code = ['#D9E500', '#E0A200', '#DC5600', '#D80C00', '#89E000', 
              '#CF007D', '#CB00BE', '#9200C7', '#4E00C3', '#0E00BF']


def plot_manifold(encoder, device, dataset, num_points, sample_dir):
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
    
def uniform(batch_size, latent_dim):
    return torch.tensor(np.random.uniform(size = (batch_size, latent_dim)),
                        dtype = torch.float)

def two_gaussian(batch_size, latent_dim):
    output = []
    for _ in range(batch_size):
        p = np.random.uniform()
        size = (batch_size, latent_dim)
        if p > 0.5:
            output.append(torch.tensor(np.random.normal(-3, 1, size = size), 
                                       dtype = torch.float).unsqueeze(0))
        else:
            output.append(torch.tensor(np.random.normal(3, 1, size = size), 
                                       dtype = torch.float).unsqueeze(0))
    output = torch.cat(output, 0)
    return output