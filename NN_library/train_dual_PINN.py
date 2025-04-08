import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *
from PDE_losses import PDE_loss_dual


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

        scheduler.step(L)
        
        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e}, mean val. rec. loss: {L_val / n_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val