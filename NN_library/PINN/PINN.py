import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)


class PeriodicLayer(nn.Module):
    def __init__(self, n, period_len):
        super().__init__()
        self.linear = nn.Linear(1, n)
        self.freq = 2*torch.pi / period_len
        self.amplitude = nn.Parameter(torch.randn(1, n))
        self.bias = nn.Parameter(torch.randn(1, n))
        
    def forward(self, x):
        n_batch = x.shape[0]
        amplitude = self.amplitude.repeat(n_batch, 1)
        bias = self.amplitude.repeat(n_batch, 1)
        x1 = self.linear(self.freq * x)
        x2 = amplitude * x1
        x3 = x2 + bias
        x4 = torch.tanh(x3)
        return x4


class ResidualLayer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, n)
        
    def forward(self, x):
        x1 = self.linear(x) + x
        x2 = torch.tanh(x1)
        return x2


class PINN(nn.Module):
    def __init__(self, n_periodic, n_hidden, n_layers, period_len):  
        super().__init__()
        self.x1_per = PeriodicLayer(n_periodic, period_len)
        self.x2_per = PeriodicLayer(n_periodic, period_len)
        self.connector = nn.Linear(2*n_periodic, n_hidden)
        layer_list = [ResidualLayer(n_hidden) for i in range(n_layers)]
        self.hidden_layers = nn.ModuleList(layer_list)
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        if len(x.shape)==1:
            x = x.unsqueeze(0)
        x1 = self.x1_per(x[:,[0]])
        x2 = self.x2_per(x[:,[1]])
        x12 = torch.cat([x1,x2], dim=1)
        x3 = torch.tanh(self.connector(x12))
        x4 = self.hidden_layers[0](x3)
        for i in range(1, len(self.hidden_layers)):
            x4 = self.hidden_layers[i](x4)
        x5 = self.output_layer(x4)
        return x5
        


