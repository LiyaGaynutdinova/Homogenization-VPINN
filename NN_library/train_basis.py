import torch
import torch.nn as nn
import torch.optim as optim
from PDE_losses import get_areas, compute_G_grad, compute_int
from save_load import *

def eig_loss(G):
    tol = 1e-7
    eigs = torch.linalg.eigvalsh(G, UPLO='U') 
    eig_penalty = torch.where(eigs<tol, eigs, 0.)
    l = eig_penalty.sum()
    return l


def train_basis(loaders, args, test_functions, n_static):

    # optimizer
    n_test = len(test_functions)
    params = []
    for i in range(n_static, n_test):
         params += test_functions[i].parameters()

    optimizer = optim.Adam(params, lr = args['lr'], maximize=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, threshold = 1e-10, verbose = True, eps=1e-10)

    losses_train = []

    for i, x in enumerate(loaders['train']):
        areas, tri = get_areas(x)
        areas = areas.to(args['dev'])
    x = x.to(args['dev']).requires_grad_(True)

    n_test = len(test_functions)
    G_static = torch.zeros((n_test, n_test), device=args['dev'])
    g_test = []

    for i in range(n_static):
        y_test = test_functions[i](x)
        grad_test = torch.autograd.grad(
                    outputs=y_test, inputs=x,
                    grad_outputs=torch.ones_like(y_test)
                )[0]
        g_test.append(grad_test.detach())

    for i in range(n_static):
        for j in range(i, n_static):
            intgr = compute_int(areas, tri, g_test[i], g_test[j]).detach()
            G_static[i,j] = intgr

    for epoch in range(args['epochs']):
        if args['dev'] == "cuda":
            torch.cuda.empty_cache() 
        G = compute_G_grad(x, areas, tri, test_functions[n_static:], g_test, G_static)
        loss_batch = eig_loss(G)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        L = loss_batch.detach().item()

        scheduler.step(L)

        losses_train.append(L)

        print(f'Epoch: {epoch} mean train loss: {L : .8e}')

        if (epoch+1)%10==0:
            for i in range(n_test):
                save_network(test_functions[i], args['name'] + f'_{i}')
    
    return losses_train