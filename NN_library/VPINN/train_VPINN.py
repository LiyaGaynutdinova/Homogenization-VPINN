import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from save_load import *
from matplotlib.tri import Triangulation


def weak_loss_avg(x, net, test_functions, a_function, H_function, G_inv, Lx):
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

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        test_prod = torch.bmm(q.view(-1,1,2), grad_test.view(-1,2,1))
        residual[i] = Lx**2 * test_prod.mean()

    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

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


def compute_int(areas, tri, fun_loc, test):
    elem = tri.triangles
    f_tri = fun_loc[elem]
    test_tri = test[elem]
    prod = torch.matmul(test_tri.view(-1,3,1,2), f_tri.view(-1,3,2,1)).squeeze()
    prod_mean = prod.mean(dim=1)
    tri_int = areas * prod_mean
    intgr = tri_int.sum()
    return intgr


def weak_loss(x, net, areas, tri, test_functions, a_function, H_function, G_inv):
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

    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)

    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        grad_test = test_functions[i]
        residual[i] = compute_int(areas, tri, q, grad_test)
   
    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L


def divergence(x, net, a_function, H_function):
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
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    # Compute divergence of q: sum of dq_i/dx_i
    divergence = torch.zeros_like(y)  # (batch_size, 1)
    for i in range(2):
        # Get q_i component (batch_size,)
        q_i = q[:, i]
        
        # Compute gradient of q_i w.r.t. x
        dq_i = torch.autograd.grad(
            outputs=q_i, inputs=x,
            grad_outputs=torch.ones_like(q_i),
            create_graph=True, retain_graph=True
        )[0]  # (batch_size, 2)
        
        # Accumulate the i-th component's derivative
        divergence += dq_i[:, i].unsqueeze(-1)
    
    return divergence


def strong_loss(x, net, test_functions, a_function, H_function, G_inv, Lx):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)

    div_u = divergence(x, net, a_function, H_function).squeeze()
    
    n_test = len(test_functions)
    residual = torch.zeros(n_test, device=x.device)

    for i in range(n_test):
        test = test_functions[i]
        test_prod = div_u * test
        residual[i] = Lx**2 * test_prod.mean()
   
    L = residual.view(1, n_test) @ G_inv @ residual.view(n_test, 1)

    return L


def PDE_loss(x, net, a_function, H_function):
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
    
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H_function(x) + grad_y  # H shape (2,) -> (batch_size, 2)
    
    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(K, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
    
    # Compute divergence of q: sum of dq_i/dx_i
    divergence = torch.zeros_like(y)  # (batch_size, 1)
    for i in range(2):
        # Get q_i component (batch_size,)
        q_i = q[:, i]
        
        # Compute gradient of q_i w.r.t. x
        dq_i = torch.autograd.grad(
            outputs=q_i, inputs=x,
            grad_outputs=torch.ones_like(q_i),
            create_graph=True, retain_graph=True
        )[0]  # (batch_size, 2)
        
        # Accumulate the i-th component's derivative
        divergence += dq_i[:, i].unsqueeze(-1)
    
    return divergence**2


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