import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from save_load import *
from matplotlib.tri import Triangulation


def weak_loss_avg(x, net, test_functions, K_loc, F_v, G_inv, Lx):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

    Q = K_loc @ grad_y.view(-1,2,1)
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        test_prod = torch.bmm(Q.view(-1,1,2), grad_test.view(-1,2,1))
        test_int = Lx**2 * test_prod.mean()
        residual[i] = test_int + F_v[i]
    
    L = residual @ G_inv @ residual.view(n_test, 1)

    return L


def get_areas(x):
    areas = []
    triang = Triangulation(x[:,0], x[:,1])
    n_elem = len(triang.triangles)
    for i in range(n_elem):
        elem = triang.triangles[i]
        node_coords_x1 = x[elem,0]
        node_coords_x2 = x[elem,1]
        vector_x1 = node_coords_x1[1:]-node_coords_x1[:-1]
        vector_x2 = node_coords_x2[1:]-node_coords_x2[:-1]
        vectors = torch.column_stack((vector_x1, vector_x2, torch.zeros((2,1))))
        area = (1/2) * abs(torch.cross(vectors[0], vectors[1])[-1])
        areas.append(area.detach())
    return torch.tensor(areas), triang


def compute_int(areas, tri, fun_loc, K_loc, test):
    elem = tri.triangles
    n_elem = len(tri.triangles)
    K_tri = K_loc[elem]
    f_tri = fun_loc[elem]
    test_tri = test[elem]
    Q = torch.matmul(K_tri, f_tri.view(n_elem, 3, 2, 1))
    prod = torch.matmul(Q.view(-1,3,1,2), test_tri.view(-1,3,2,1)).squeeze()
    prod_mean = prod.mean(dim=1)
    tri_int = areas * prod_mean
    intgr = tri_int.sum()
    return intgr


def weak_loss(x, net, areas, tri, test_functions, K_loc, F_v, G_inv):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    y = net(x)
    
    # Compute ∇y: (batch_size, 2)
    grad_y = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

    Q = K_loc @ grad_y.view(-1,2,1)
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        test_int = compute_int(areas, tri, Q, K_loc, grad_test)
        residual[i] = test_int + F_v[i]
    
    L = residual @ G_inv @ residual.view(n_test, 1)

    return L





def train(net, loaders, args, a_function, H, test_functions, G, Lx, int_type='trap'):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])

    losses_train = []

    # prepare test functions
    n_test = len(test_functions)
    g_test = []
    F_v = []
    for i, x in enumerate(loaders['train']):
        if int_type=='trap':
            areas, tri = get_areas(x)
            areas = areas.to(args['dev'])
        x = x.to(args['dev']).clone().detach().requires_grad_(True)
        K_loc = a_function(x).detach()
        H_loc = H(x).detach()
        for i in range(n_test):
            y_test = test_functions[i](x)
            grad_test = torch.autograd.grad(
                outputs=y_test, inputs=x,
                grad_outputs=torch.ones_like(y_test)
            )[0]
            g_test.append(grad_test.detach())
            if int_type=='trap':
                F_v.append(compute_int(areas, tri, H_loc, K_loc, grad_test))
            else:
                f_loc = H_loc.view(-1,1,2) @ K_loc @ grad_test.view(-1,2,1)
                F_v.append(f_loc.mean().detach()*Lx**2)


    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        L = 0
        net.train()
        for i, x in enumerate(loaders['train']):
            x = x.to(args['dev'])
            if int_type=='trap':
                l = weak_loss(x, net, areas, tri, g_test, K_loc, F_v, G)
            else:
                l = weak_loss_avg(x, net, g_test, K_loc, F_v, G, Lx)
            loss_batch = l.detach().item()
            L += loss_batch
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        
        losses_train.append(L)

        print(f'Epoch: {epoch} mean train loss: {L : .8e}')#, mean val. rec. loss: {L_val / n_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train