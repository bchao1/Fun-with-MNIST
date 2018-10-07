# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:06:07 2018

@author: USER
"""
import torch
import torch.optim as optim
from torch.autograd import grad, Variable
import numpy as np

def get_optim(model):
    return optim.Adam(model.parameters(), betas = (0.5, 0.999), lr = 0.0002)

def interpolate(real_img, fake_img):
    N = real_img.shape[0]
    theta = torch.tensor(np.random.uniform(size = N), dtype = torch.float).view(N, 1, 1, 1).cuda()
    sample = theta * real_img + (1 - theta) * fake_img
    return sample

def gradient_norm(model, real_img, fake_img):
    N = real_img.shape[0]
    _input = interpolate(real_img, fake_img)
    _input = Variable(_input, requires_grad = True)
    score = model(_input)
    outputs = torch.zeros(score.shape).cuda()
    gradient = grad(outputs = score, 
                    inputs = _input, 
                    grad_outputs = outputs,
                    create_graph = True,
                    retain_graph = True,
                    only_inputs = True)[0]
    grad_norm = (gradient.view(N, -1).norm(p = 2) - 1)**2
    return grad_norm