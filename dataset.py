
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, size, x, y):
        torch.manual_seed(0)
        self.data = torch.rand(size, 2)
        self.data[:, 0] = x[0] + (x[1] - x[0]) * self.data[:, 0]
        self.data[:, 1] = y[0] + (y[1] - y[0]) * self.data[:, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xy = self.data[idx]
        return xy


def get_loaders(data, batch_size):
    n_train = int(0.8 * data.__len__())
    n_test = (data.__len__() - n_train) // 2
    n_val = data.__len__() - n_train - n_test
    torch.manual_seed(0)
    train_set, val_set, test_set = torch.utils.data.random_split(data, [n_train, n_val, n_test])
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    return loaders
