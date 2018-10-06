# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 21:28:25 2018

@author: USER
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Classifier import Net
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_onehot(labels, classes):
    batch_size = labels.shape[0]
    onehot = torch.zeros(batch_size, classes)
    for i in range(batch_size):
        onehot[i][labels[i]] = 1
    return onehot

def accuracy(model, dataloader):
    correct = 0
    for i, (img, label) in enumerate(dataloader):
        img = img.to(device)
        label = label.to(device)
        predict_prob = model(img)
        predict_label = torch.max(predict_prob, 1)[1]
        correct += sum(label == predict_label).item()
    return correct / dataloader.dataset.__len__()



batch_size = 128
epochs = 100
num_classes = 10
transform = transforms.Compose([
            transforms.Pad(padding = 2, padding_mode = 'edge'),
            transforms.ToTensor()
            ])

train_data = MNIST(root = '../data', train = True, download = False, 
                   transform = transform)
train_loader = DataLoader(dataset = train_data, batch_size = batch_size,
                          shuffle = True)

test_data = MNIST(root = '../data', train = False, download = False, 
                   transform = transform)
test_loader = DataLoader(dataset = test_data, batch_size = batch_size,
                          shuffle = True)

classifier = Net(num_classes).to(device)
optimizer = optim.Adam(classifier.parameters(), betas = (0.5, 0.999), lr = 0.0002)

criterion = nn.BCELoss().to(device)

train_loss = []

train_accuracy = []
test_accuracy = []

steps_per_epoch = int(np.ceil(train_data.__len__() / batch_size))

for epoch_i in range(1, epochs + 1):
    running_loss = 0
    for i, (img, label) in enumerate(train_loader):
        
        img = img.to(device)
        onehot_label = to_onehot(label, num_classes).to(device)
        
        predict = classifier(img)
        loss = criterion(predict, onehot_label)
        running_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch [{}], Process [{}/{}], Loss [{:8f}]'.
              format(epoch_i, i + 1, steps_per_epoch, loss.item()))
        
        
    running_loss /= steps_per_epoch
    train_loss.append(running_loss.item())
    
    train_accuracy.append(accuracy(classifier, train_loader))
    test_accuracy.append(accuracy(classifier, test_loader))    
    
    print('========== Train accuracy [{:8f}] =========='.format(train_accuracy[-1]))
    print('========== Test accuracy [{:8f}] =========='.format(test_accuracy[-1]))
    
    x = list(range(1, epoch_i + 1))
    plt.plot(x, train_accuracy)
    plt.plot(x, test_accuracy)
    plt.savefig('accuracy.png')
    plt.close()
    plt.plot(x, train_loss)
    plt.savefig('loss.png')
    plt.close()
    

