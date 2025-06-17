import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *
from PDE_losses import PDE_loss_dual, get_areas, compute_estimate


def train_dual(net, loaders, args, a_function, H, Lx):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, threshold = 1e-10, verbose = True, eps=1e-10)


    losses_train = []
    losses_val = []

    for i, x in enumerate(loaders['train']):
            areas, tri = get_areas(x)
            areas = areas.to(args['dev'])
    x = x.to(args['dev'])

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        L = 0
        net.train()
        l, q, gH = PDE_loss_dual(x, net, a_function, H)
        bound = compute_estimate(areas, tri, q, gH, Lx).detach()
        bound_inv = 1 / bound[0]
        loss_batch = l.mean()
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        L += loss_batch.detach().item()

        scheduler.step(L)
        
        losses_train.append(L)
        losses_val.append(bound_inv.item())

        print(f'Epoch: {epoch} mean train loss: {L : .8e}, bound: {bound_inv.item() : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val