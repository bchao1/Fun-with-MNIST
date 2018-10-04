# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:06:07 2018

@author: USER
"""
from torch.autograd import grad, Variable
import numpy as np

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