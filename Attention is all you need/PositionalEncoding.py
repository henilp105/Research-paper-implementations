import torch
import torch.nn as nn   
from Block import Block

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformers
    """
    def __init__(self,n_embed,block_size,device):
        super().__init__()
        self.n_embed = n_embed
        self.context_length = block_size
        self.device = device

    def forward(self,x):
        pos_encoding = torch.zeros(self.context_length,self.n_embed,device=self.device)
        position = torch.arange(0,self.context_length,device=self.device).float().unsqueeze(dim=1)
        i = torch.arange(0,self.n_embed,step=2,device=self.device).float()
        pos_encoding[:,0::2] = torch.sin(position / (1000 ** (i/ self.n_embed)))
        pos_encoding[:,1::2] = torch.cos(position / (1000 ** (i/ self.n_embed)))
        batch_size, seq_length = x.size()
        return pos_encoding[:seq_length, :].expand(batch_size,seq_length, self.n_embed)
        