import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PDE_losses import *
from save_load import *


def train_dual(net, loaders, args, a_function, H, test_functions, G, Lx, int_type='trap'):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, cooldown=100, threshold = 1e-10, verbose = True, eps=1e-10)

    losses_train = []
    losses_val = []

    # prepare test functions
    n_test = len(test_functions)
    g_test = []
    for i, x in enumerate(loaders['train']):
        if int_type=='trap':
            areas, tri = get_areas(x)
            areas = areas.to(args['dev'])
        x = x.to(args['dev']).clone().detach().requires_grad_(True)
        for i in range(n_test):
            y_test = test_functions[i](x)
            grad_test = torch.autograd.grad(
                outputs=y_test, inputs=x,
                grad_outputs=torch.ones_like(y_test)
            )[0]
            curl_test = torch.zeros_like(grad_test)
            curl_test[:,0] = -grad_test[:,1]
            curl_test[:,1] = grad_test[:,0]
            g_test.append(curl_test.detach())

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        net.train()
        L, q, gH = weak_loss_dual(x, net, areas, tri, g_test, a_function, H, G)
        bound = compute_estimate(areas, tri, q, gH, Lx).detach()
        bound_inv = 1 / bound
        optimizer.zero_grad()
        L.backward()
        optimizer.step()

        scheduler.step(L)
        
        losses_train.append(L.detach().item())
        losses_val.append(bound_inv.item())


        print(f'Epoch: {epoch} mean train loss: {L.detach().item() : .8e}, mean val. loss: {bound_inv.item() : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val