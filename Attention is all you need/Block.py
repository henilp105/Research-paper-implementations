import torch
import torch.nn as nn
from MHSAttention import MHSAttention


class FeedForward(nn.Module):
    """
    FeedForward layers with non linearity
    """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    """
    Simple Transformer Block
    """

    def __init__(self, n_embed, num_heads, block_size, dropout):
        super().__init__()
        self.MHSAttention = MHSAttention(
            num_heads, n_embed, block_size, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.MHSAttention(x)
        x = self.layer_norm1(x)
        x = x + self.ff(x)
        x = self.layer_norm2(x)
        return x


n_embed = 128  # embedding size, C
block_size = 128  # context length, B
dropout = 0.0
num_heads = 4
head_size = n_embed // num_heads
num_queries = 100  # number of queries, T

x = torch.randn(block_size, num_queries, n_embed)
TransformerBlock = Block(n_embed, num_heads, block_size, dropout)
print(TransformerBlock(x))