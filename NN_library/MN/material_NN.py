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


class Material_NN(nn.Module):
    def __init__(self, period_len=2*torch.pi):  
        super().__init__()

        self.x1_per = PeriodicLayer(1, period_len)
        self.x2_per = PeriodicLayer(1, period_len)

        self.linear = nn.Sequential(
            nn.Linear(2,2),
            nn.Softmax(dim=-1)
            )

    def forward(self, x):
        x1 = self.x1_per(x[:,[0]])
        x2 = self.x2_per(x[:,[1]])
        x12 = torch.cat([x1, x2], dim=1)
        y = self.linear(x12)
        return y[:,[0]]
