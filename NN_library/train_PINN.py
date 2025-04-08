import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *


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



def train(net, loaders, args, a_function, H, v_function=None):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=40, threshold = 1e-10, verbose = True, eps=1e-10)

    n_train = len(loaders['train'].dataset)
    n_val = len(loaders['val'].dataset)

    losses_train = []
    losses_val = []

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        L = 0
        net.train()
        for i, x in enumerate(loaders['train']):
            x = x.to(args['dev'])
            #x.requires_grad_()
            #y = net(x)
            l = PDE_loss(x, net, a_function, H).sum()
            loss_batch = l.detach().item()
            L += loss_batch
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()

        # calculate the loss and accuracy of the validation set
        net.eval()
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        
        L_val = 0
        for j, x_val in enumerate(loaders['val']):
            x_val = x_val.to(args['dev'])
            #x_val.requires_grad_()
            #y_val = net(x_val)
            L_val += PDE_loss(x_val, net, a_function, H).detach().sum().item()
            
        #scheduler.step(L)

        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e}, mean val. loss: {L_val / n_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val