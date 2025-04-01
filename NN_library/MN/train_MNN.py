import torch
import torch.nn as nn
import torch.optim as optim
from save_load import *


def train(net, loaders, args, a_function):

    # network
    net.to(args['dev'])

    loss = nn.BCELoss()

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
            y = net(x)
            l = loss(y, a_function(x).detach()).sum()
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
            y_val = net(x_val)
            L_val += loss(y_val, a_function(x_val).detach()).detach().sum().item()
            
        #scheduler.step(L_val)

        losses_train.append(L / n_train)
        losses_val.append(L_val / n_val)

        if (epoch+1)%100 == 0:
            print(f'Epoch: {epoch} mean train loss: {L / n_train : .8e}, mean val. rec. loss: {L_val / n_val : .8e}')
        if (epoch+1)%1000 == 0:
            save_network(net, args['name'] + f'_{epoch}')
    
    return losses_train, losses_val