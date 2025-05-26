import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PDE_losses import *
from save_load import *


def train(net, loaders, args, a_function, H, test_functions, G, Lx, rho=0.01):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1000, threshold = 1e-10, verbose = True, eps=1e-10)

    loss_bound = nn.MSELoss(reduction='none')

    losses_train = []
    losses_val = []

    # prepare test functions
    n_test = len(test_functions)
    g_test = []
    
    curl_test = []
    for i, x in enumerate(loaders['train']):
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
            curl = torch.zeros_like(grad_test)
            curl[:,0] = -grad_test[:,1]
            curl[:,1] = grad_test[:,0]
            curl_test.append(curl.detach())


    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        L = 0
        net.train()
        pde_p, pde_d, q_p, q_d = weak_loss_primal_dual(x, net, areas, tri, g_test, curl_test, a_function, H, G)
        bound_p = compute_bound(areas, tri, q_p, Lx)
        bound_d = compute_bound(areas, tri, q_d, Lx)
        dot_prod_1 = torch.dot(bound_p, bound_d)
        dot_prod_2 = torch.dot(bound_p, torch.flip(bound_d, dims=(0,)))
        l_bound_1 = (dot_prod_1-1.)**2
        l_bound_2 = dot_prod_2**2
        l = pde_p.mean() + pde_d.mean() + rho*torch.sqrt(2*l_bound_1 + 2*l_bound_2)
        loss_batch = l.detach().item()
        L += loss_batch
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        #net.eval()
        #if args['dev'] == "cuda":
            #torch.cuda.empty_cache() 
        
        #L_val = 0
        #for j, x_val in enumerate(loaders['val']):
            #x_val = x_val.to(args['dev'])
            #pde_p, pde_d, _, _ = PDE_loss_primal_dual(x, net, a_function, H)
            #L_val += (pde_p + pde_d).detach().mean().item()

        scheduler.step(L)

        losses_train.append(L)
        losses_val.append(dot_prod_1.item())


        print(f'Epoch: {epoch} mean train loss: {L : .8e}, mean val. loss: {dot_prod_1.item() : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val