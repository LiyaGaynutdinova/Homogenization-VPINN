import torch
import torch.nn as nn

def save_network(net:nn.Module, filename):
    torch.save({'model_state_dict': net.state_dict()}, filename)

def load_network(net, filename, args):
    f = open(filename, "rb")
    state = torch.load(f , map_location=args['dev'])
    net.load_state_dict(state['model_state_dict'])
    f.close()
    return net