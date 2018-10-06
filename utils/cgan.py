# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:56:54 2018

@author: USER
"""

import torch
import numpy as np

def to_onehot(labels, classes):
    batch_size = labels.shape[0]
    onehot = torch.zeros(batch_size, classes)
    for i in range(batch_size):
        onehot[i][labels[i]] = 1
    return onehot


def get_label_mismatch(class_label, num_classes):
    batch_size = class_label.shape[0]
    mismatch = torch.zeros(batch_size)
    for i in range(batch_size):
        shift = np.random.randint(1, num_classes)
        mismatch[i] = (class_label[i] + shift) % num_classes
    return torch.tensor(mismatch, dtype = torch.long)