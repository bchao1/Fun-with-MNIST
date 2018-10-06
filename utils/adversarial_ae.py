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

color_code = np.array(['#D9E500', '#E0A200', '#DC5600', '#D80C00', '#89E000', 
              '#CF007D', '#CB00BE', '#9200C7', '#4E00C3', '#0E00BF'])

def to_onehot(labels, classes):
    batch_size = labels.shape[0]
    onehot = torch.zeros(batch_size, classes)
    for i in range(batch_size):
        onehot[i][labels[i]] = 1
    return onehot

def plot_manifold(encoder, device, dataset, num_points, sample_dir):
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.figure(figsize=(6, 6), dpi=100)
    labels = []
    data = []
    for i in range(num_points):
        img, label = dataset.__getitem__(i)
        img = img.view(-1).unsqueeze(0).to(device)
        labels.append(label)
        data.append(img)
    data = torch.cat(data, 0)
    labels = np.array(labels)
    code = encoder(data).squeeze()
    x, y = code.detach().cpu().numpy().transpose()
    plt.scatter(x[i], y[i], marker = '.', c = color_code[labels])
    
    patches = []
    for i in range(len(color_code)):
        patches.append(mpatches.Patch(color=color_code[i], label=i))
    plt.legend(handles = patches)
    plt.savefig(os.path.join(sample_dir, 'manifold_scatter.png'))
    
def uniform(batch_size, latent_dim):
    return torch.tensor(np.random.uniform(size = (batch_size, latent_dim)),
                        dtype = torch.float)

def two_gaussian(batch_size, latent_dim = 2, center = 5):
    
    output = list(range(batch_size))
    def get_point(dummy):
        p = np.random.uniform()
        if p > 0.5:
            return torch.tensor(np.random.normal(-center, 1, latent_dim), 
                                       dtype = torch.float).unsqueeze(0)
        return torch.tensor(np.random.normal(center, 1, latent_dim), 
                                       dtype = torch.float).unsqueeze(0)
        
    output = list(map(get_point, output))
    return torch.cat(output, 0)


def triangle(batch_size):
    
    A = np.array([0, 1])
    B = np.array([np.sqrt(3) / 2, -0.5])
    C = np.array([-np.sqrt(3) / 2, -0.5])
    
    p = list(range(batch_size))
    def get_point(dummy):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        sample = ((1 - np.sqrt(r1)) * A + 
                  (np.sqrt(r1) * (1 - r2)) * B +
                  (r2 * np.sqrt(r1)) * C)
        return torch.tensor(sample, dtype = torch.float).unsqueeze(0)
    
    p = list(map(get_point, p))
    return torch.cat(p, 0)

def donut(batch_size, r1 = 1, r2 = 2):
    p = list(range(batch_size))
    
    def get_point(dummy):
        u = np.random.uniform()
        r = np.sqrt(u*(r1**2) + (1-u)*(r2**2)) 
        theta = np.random.uniform(0, 360)
        coor = [r * np.sin(theta), r*np.cos(theta)]
        return torch.tensor(coor, dtype = torch.float).unsqueeze(0)
    
    p = list(map(get_point, p))
    return torch.cat(p, 0)