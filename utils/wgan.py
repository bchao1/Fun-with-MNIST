# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:05:28 2018

@author: USER
"""

def clip_weights(model, c):
    for p in model.parameters():
        p.data.clamp_(-c, c)
    return