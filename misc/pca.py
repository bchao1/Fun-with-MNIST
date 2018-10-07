# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:35:59 2018

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

color_code = np.array(['#D9E500', '#E0A200', '#DC5600', '#D80C00', '#89E000', 
              '#CF007D', '#CB00BE', '#9200C7', '#4E00C3', '#0E00BF'])

def get_data():
    return MNIST(root = '../data', train = True, 
                 transform = transforms.ToTensor(), download = False)

def pack(data):
     idx = list(range(len(data)))
     def get_img(i):
         return data.__getitem__(i)[0].view(-1).unsqueeze(0).numpy()
     def get_label(i):
         return data.__getitem__(i)[1].view(-1).unsqueeze(0).numpy()
     
     batch = list(map(get_img, idx))
     labels = list(map(get_label, idx))
     return np.concatenate(batch, 0), np.concatenate(labels, 0)

def reduction(data):
    batch, label = pack(data)
    pca = PCA(n_components = 2)
    reduce = pca.fit_transform(batch).transpose()
    x, y = reduce
    return x, y, label.transpose().flatten()

def plot(x, y, labels):
    plt.gca().set_aspect('equal', adjustable='box')
    plt.figure(figsize=(6, 6), dpi=100)
    
    plt.scatter(x, y, marker = '.', c = color_code[labels])
    color_idx = list(range(len(color_code)))
    color_legend = list(map(lambda i:patches.Patch(color = color_code[i], 
                                                   label = i), color_idx))
    plt.legend(handles = color_legend)
    plt.savefig('mnist_pca.png')

def main():
    data = get_data()
    batch, labels = pack(data)
    x, y = reduction(batch)
    plot(x, y, labels)

if __name__ == '__main__':
    main()
