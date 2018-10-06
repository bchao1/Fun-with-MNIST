# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:22:33 2018

@author: USER
"""

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
                )
        self.classifier = nn.Sequential(
                nn.Linear(2048, 10),
                nn.BatchNorm1d(10),
                nn.Softmax(dim = 1)
                )
        
    def forward(self, input):
        N = input.shape[0]
        features = self.feature_extractor(input).view(N, -1)
        return self.classifier(features)