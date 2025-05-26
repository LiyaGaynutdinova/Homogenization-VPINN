import torch
import torch.nn as nn
import torch.optim as optim
from PDE_losses import PDE_loss_primal_dual, compute_bound, get_areas
from save_load import *


def train(net, loaders, args, a_function, H, Lx, rho=0.001):

    # network
    net.to(args['dev'])

    # optimizer
    optimizer = optim.Adam(net.parameters(), lr = args['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=40, threshold = 1e-10, verbose = True, eps=1e-10)

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
        pde_p, pde_d, q_p, q_d = PDE_loss_primal_dual(x, net, a_function, H)
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

        # calculate the loss and accuracy of the validation set
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