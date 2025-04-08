import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PDE_losses import *
from save_load import *


def train(net, loaders, args, a_function, H, test_functions, G, Lx, int_type='trap'):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])

    losses_train = []
    losses_train_ref = []
    losses_val = []

    # prepare test functions
    n_test = len(test_functions)
    test_f = []
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
            g_test.append(grad_test.detach())
            test_f.append(y_test.detach())


    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        net.train()
        for i, x in enumerate(loaders['train']):
            x = x.to(args['dev'])
            L = weak_loss(x, net, areas, tri, g_test, a_function, H, G)
            #L = strong_loss(x, net, test_f, a_function, H, G, Lx)
            #L_ref = weak_loss_avg(x, net, g_test, a_function, H, G, Lx)
            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0
        for j, x_val in enumerate(loaders['val']):
            x_val = x_val.to(args['dev'])
            L_val += PDE_loss(x_val, net, a_function, H).detach().mean().item()
        
        losses_train.append(L.detach().item())
        #losses_train_ref.append(L_ref.detach().item())
        losses_val.append(L_val)


        print(f'Epoch: {epoch} mean train loss: {L.detach().item() : .8e}, mean val. loss: {L_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_train_ref, losses_val