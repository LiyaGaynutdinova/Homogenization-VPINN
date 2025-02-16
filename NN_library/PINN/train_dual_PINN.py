import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *


def PDE_loss_dual(x, net, a_function, H):
    # Ensure x has requires_grad; clone to avoid modifying original tensor
    x = x.clone().detach().requires_grad_(True)
    
    # Forward pass to get y (batch_size, 1)
    u = net(x)
    
    # Compute ∇y: (batch_size, 2)
    du = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    curl_u = torch.zeros_like(du)
    curl_u[:,0] = -du[:,1]
    curl_u[:,1] = du[:,0]
   
    # H + ∇y (broadcast H to batch_size, 2)
    H_plus_grad = H + curl_u  # H shape (2,) -> (batch_size, 2)
   
    # Compute K(x) as (batch_size, 2, 2)
    R = a_function(x)
    
    # Compute q = K @ (H + ∇y) using batched matrix multiplication
    q = torch.bmm(R, H_plus_grad.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2)
   
    # Compute gradient of q_1 w.r.t. x
    dq1 = torch.autograd.grad(
        outputs=q[:,0], inputs=x,
        grad_outputs=torch.ones_like(q[:,0]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    # Compute gradient of q_2 w.r.t. x
    dq2 = torch.autograd.grad(
        outputs=q[:,1], inputs=x,
        grad_outputs=torch.ones_like(q[:,1]),
        create_graph=True, retain_graph=True
    )[0]  # (batch_size, 2)

    curl = dq2[:,0] - dq1[:,1]
    
    return curl**2


def train(net, loaders, args, a_function, H):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, threshold = 1e-10, verbose = True, eps=1e-10)

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

        scheduler.step(L_val)
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e}, mean val. rec. loss: {L_val / n_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val