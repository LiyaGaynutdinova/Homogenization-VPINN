import torch
import torch.nn as nn
import torch.nn.functional as F

# query if we have GPU
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', dev)


class PeriodicLayer(nn.Module):
    def __init__(self, n, period_len):
        super().__init__()
        self.linear = nn.Linear(1, n)
        self.freq = 2*torch.pi / period_len
        self.amplitude = nn.Parameter(torch.randn(1, n))
        self.bias1 = nn.Parameter(torch.randn(1, n))
        self.bias2 = nn.Parameter(torch.randn(1, n))
        
    def forward(self, x):
        n_batch = x.shape[0]
        amplitude = self.amplitude.repeat(n_batch, 1)
        bias1 = self.bias1.repeat(n_batch, 1)
        bias2 = self.bias2.repeat(n_batch, 1)
        x1 = amplitude * torch.cos(self.freq * x + bias1)
        x2 = torch.tanh(x1 + bias2)
        return x2


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
        layer_list = []
        for i in range(n_layers):
            layer_list.append(ResidualLayer(n_hidden))
        self.hidden_layers = nn.ModuleList(layer_list)
        self.output_layer = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x1 = self.x1_per(x[:,[0]])
        x2 = self.x2_per(x[:,[1]])
        x12 = torch.cat([x1,x2], dim=1)
        x3 = torch.tanh(self.connector(x12))
        x4 = self.hidden_layers[0](x3)
        for i in range(len(self.hidden_layers)):
            x4 = self.hidden_layers[i](x4)
        x5 = self.output_layer(x4)
        x6 = x5 - x5.mean().detach()
        return x6
