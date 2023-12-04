from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data.sampler import SubsetRandomSampler

class Predictor(nn.Module):
    r"""Predictor. 
    
    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self):
        super().__init__()


    def forward(self, h1, h2):
        with torch.no_grad():
            h2_mean = h2 - h2.mean(dim=0)
            h2_mean = F.normalize(h2_mean, dim=1)
            w = torch.mm(h2_mean.t(), h2_mean) / (h2.size(0) - 1)  
        return torch.mm(h1, w.detach())
