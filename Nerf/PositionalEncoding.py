import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, L=10):
        super(PositionalEncoding, self).__init__()
        self.L = L  

    def forward(self, x):
        freq_bands = 2.0 ** torch.arange(self.L)  
        freq_bands = freq_bands.to(x.device) * torch.pi  

        sin_terms = torch.sin(x[..., None] * freq_bands)  
        cos_terms = torch.cos(x[..., None] * freq_bands) 

        return torch.cat([sin_terms, cos_terms], dim=-1).view(x.shape[0], -1)  

