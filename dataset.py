
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats.qmc import Sobol

class dataset(Dataset):
    def __init__(self, size, x, y):
        torch.manual_seed(0)
        self.data = torch.rand(size, 2)
        self.data[:, 0] = x[0] + (x[1] - x[0]) * self.data[:, 0]
        self.data[:, 1] = y[0] + (y[1] - y[0]) * self.data[:, 1]
        boundary_size = int(0.01 * size)
        self.data[:boundary_size, 0] = x[0]
        self.data[boundary_size:boundary_size*2, 0] = x[1]
        self.data[boundary_size:boundary_size*2, 1] = self.data[:boundary_size, 1]
        self.data[boundary_size*2:boundary_size*3, 1] = y[0]
        self.data[boundary_size*3:boundary_size*4, 1] = y[1]
        self.data[boundary_size*3:boundary_size*4, 0] = self.data[boundary_size*2:boundary_size*3, 0]
        self.data[0] = torch.tensor([x[0],y[0]])
        self.data[1] = torch.tensor([x[0],y[1]])
        self.data[2] = torch.tensor([x[1],y[0]])
        self.data[3] = torch.tensor([x[1],y[1]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xy = self.data[idx]
        return xy
    

class dataset_Sobol(Dataset):
    def __init__(self, size, x, y, seed=0):
        sampler = Sobol(d=2, scramble=True, seed=seed)
        data = sampler.random_base2(m=size)
        tol = 1 / (2**size - 1)
        data[:, 0] = np.where(data[:, 0]<tol, 0, data[:, 0])
        data[:, 0] = np.where(data[:, 0]>(1-tol), 1, data[:, 0])
        data[:, 1] = np.where(data[:, 1]<tol, 0, data[:, 1])
        data[:, 1] = np.where(data[:, 1]>(1-tol), 1, data[:, 1])
        corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        data = np.vstack((data, corners))
        data[:, 0] = x[0] + (x[1] - x[0]) * data[:, 0]
        data[:, 1] = y[0] + (y[1] - y[0]) * data[:, 1]
        self.data = torch.tensor(data, dtype=torch.float)
        self.x = x
        self.y = y
        self.size = size


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xy = self.data[idx]
        return xy

class dataset_grid(Dataset):
    def __init__(self, N, x, y):
        self.x = x
        self.y = y
        x_n = np.linspace(x[0], x[1], N, endpoint=True)
        y_n = np.linspace(y[0], y[1], N, endpoint=True)
        XY = np.meshgrid(x_n, y_n)
        data = np.vstack((XY[0].flatten(), XY[1].flatten())).T
        self.data = torch.tensor(data, dtype=torch.float)


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
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size = batch_size)
    test_loader = DataLoader(test_set, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader, 'test' : test_loader}
    return loaders


def get_loaders_Sobol(data, batch_size):
    n_train = data.__len__()
    N = int(np.sqrt(0.2 * n_train))
    x = np.linspace(data.x[0], data.x[1], N+1, endpoint=False)
    y = np.linspace(data.y[0], data.y[1], N+1, endpoint=False)
    XY = np.meshgrid(x, y)
    data_val = np.vstack((XY[0].flatten(), XY[1].flatten())).T
    data_val = torch.tensor(data_val, dtype=torch.float)
    train_loader = DataLoader(data, batch_size = batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(data_val, batch_size = batch_size)
    loaders = {'train' : train_loader, 'val' : val_loader}
    return loaders
