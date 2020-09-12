import torch
from torch import nn, optim
from torch.nn import functional as F

def mlp(in_features=3, h_dims=[256,256], actv=nn.ReLU, out_actv=nn.ReLU,
        USE_DROPOUT=False, device=None):
    """
    Multi-layer perceptron 
    """
    layers = []
    in_features = in_features
    for h_dim in h_dims[:-1]:
        
        ki = torch.randn(h_dim, in_features, dtype=torch.float, requires_grad=True)
        
        linear = nn.Linear(in_features, h_dim)
        linear.weight = nn.Parameter(ki)
        layers.append(linear)
        
        act = actv(inplace=True).to(device)
        layers.append(act)

        in_features = h_dim
        
        if USE_DROPOUT:
            layers.append(nn.Dropout())
                
    ki = torch.randn(h_dims[-1], in_features, dtype=torch.float, requires_grad=True)
    linear = nn.Linear(in_features, h_dims[-1])
    linear.weight = nn.Parameter(ki)
    layers.append(linear)
    
    if out_actv:
        act = out_actv(inplace=True)
        layers.append(act)
    
    return nn.Sequential(*layers).to(device)