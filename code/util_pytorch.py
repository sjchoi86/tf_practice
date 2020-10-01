import torch
from torch import nn, optim
from torch.nn import functional as F
from scipy.stats import truncnorm


def truncated_normal(size, threshold=1, dtype=torch.float, requires_grad=True):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    values = torch.from_numpy(values).type(dtype)
    values.requires_grad = requires_grad
    return values

class mlp(nn.Module):
    def __init__(self, in_features=784, h_dims=[256, 256], 
                    actv=nn.ReLU, out_actv=nn.ReLU, USE_DROPOUT=False):
        """
        Multi-layer perceptron 
        """
        super(mlp, self).__init__()

        layers = []
        in_features = in_features
        for h_dim in h_dims[:-1]:

            linear = nn.Linear(in_features, h_dim)
            # ki = truncated_normal(size=(h_dim, in_features), dtype=torch.float, requires_grad=True)
            # linear.weight = nn.Parameter(ki)
            layers.append(linear)
            
            act = actv(inplace=True)
            layers.append(act)

            in_features = h_dim
            
            if USE_DROPOUT:
                layers.append(nn.Dropout())
        linear = nn.Linear(in_features, h_dims[-1])     
        # ki = truncated_normal(size=(h_dims[-1], in_features), dtype=torch.float, requires_grad=True)
        # linear.weight = nn.Parameter(ki)
        layers.append(linear)
        
        if out_actv:
            act = out_actv(inplace=True)
            layers.append(act)
        
        self.moedl = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.moedl(X)