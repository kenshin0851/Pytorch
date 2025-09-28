# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 21:47:26 2025

@author: kimke
"""

import torch
import torch.nn as nn

x = torch.tensor([[1.0,-1.0],
                  [0.0,1.0],
                  [0.0, 0.0]])

in_features = x.shape[1]
out_features =2 

m =nn.Linear(in_features,out_features)
m_weight = m.weight
m_bias = m.bias

y = m(x)

n =nn.ReLU()
output = n(x)