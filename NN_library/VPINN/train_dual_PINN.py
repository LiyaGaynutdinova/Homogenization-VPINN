import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *


def PDE_loss_dual(x, net, a_function, H):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get q (batch_size, 1)
    q_tilde = net(x)

    # Compute K(x) as (batch_size, 2, 2)
    K = a_function(x)
    KH = torch.matmul(K, H.view(1,2,1)).squeeze(-1)    # H shape (2,) -> (batch_size, 2)

    # q_tilde - KH (broadcast H to batch_size, 2)
    q = q_tilde - KH
 
    # Compute ∇q: (batch_size, 2)
    divergence = torch.zeros_like(q_tilde[:,0])  # (batch_size, 1)
    curl = torch.zeros_like(q_tilde[:,0])  # (batch_size, 1)
    
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
        divergence += dq_i[:, i]

        # Get q_tilde_i component (batch_size,)
        q_tilde_i = q_tilde[:, i]
        
        # Compute gradient of q_tilde_i w.r.t. x
        dq_tilde_i = torch.autograd.grad(
            outputs=q_tilde_i, inputs=x,
            grad_outputs=torch.ones_like(q_i),
            create_graph=True, retain_graph=True
        )[0]  # (batch_size, 2)
        
        # Accumulate the i-th component's derivative
        curl += (-1)**i * dq_tilde_i[:, i]

    return divergence**2 + curl**2



def train(net, loaders, args, a_function, H):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    #optimizer = optim.LBFGS(net.parameters(), lr = args['lr'])

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
            l = PDE_loss_dual(x, net, a_function, H).sum()
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
            L_val += PDE_loss_dual(x_val, net, a_function, H).detach().sum().item()
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e}, mean val. rec. loss: {L_val / n_val : .8e}')
        if (epoch+1)%100 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val