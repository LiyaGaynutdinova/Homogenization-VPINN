import torch
import torch.nn as nn
import torch.optim as optim
from PDE_losses import get_areas, compute_G
from save_load import *


def train_basis(loaders, args, test_functions):

    # optimizer
    n_test = len(test_functions)
    params = []
    for i in range(n_test):
         params += test_functions[i].parameters()

    optimizer = optim.Adam(params, lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, threshold = 1e-10, verbose = True, eps=1e-10)

    losses_train = []

    for i, x in enumerate(loaders['train']):
            areas, tri = get_areas(x)
            areas = areas.to(args['dev'])
    x = x.to(args['dev'])
    I = torch.eye(n_test, device=args['dev'])

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        G = compute_G(x, areas, tri, test_functions)
        loss_batch = torch.linalg.matrix_norm(G-I, ord='fro')
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        L = loss_batch.detach().item()

        scheduler.step(L)

        losses_train.append(L)

        print(f'Epoch: {epoch} mean train loss: {L : .8e}')

    if (epoch+1)%10==0:
        for i in range(n_test):
            save_network(test_functions[i], args['name'] + f'_{i}')
    
    return losses_train