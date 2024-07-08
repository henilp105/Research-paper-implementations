import torch
import torch.nn as nn
from Block import Block
from PositionalEncoding import PositionalEncoding


class Encoder(nn.Module):
    """
    Encoder Block for a transformer
    """

    def __init__(self, vocab_size, num_layers, n_embed, num_heads, block_size, dropout, device):
        self.word_embedding = nn.Embedding(vocab_size, n_embed)
        self.block = nn.ModuleList(
            [Block(n_embed, num_heads, block_size, dropout) for _ in num_layers])
        self.pos_encoder = PositionalEncoding(n_embed, block_size, device)

    def forward(self, x):
        x = self.word_embedding(x) + self.pos_encoder(x)
        for layer in self.block:
            x = layer(x)
        return x

