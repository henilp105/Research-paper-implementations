import torch
import torch.nn as nn
from Block import Block
from PositionalEncoding import PositionalEncoding
from MHSAttention import MHSAttention


class DecoderBlock(nn.Module):
    """
    Decoder Block for a transformer
    """

    def __init__(self, n_embed, num_heads, block_size, dropout):
        self.block = Block(n_embed, num_heads, block_size, dropout)
        self.MHSAttention = MHSAttention(num_heads, n_embed, block_size, dropout)
        self.layer_norm = nn.LayerNorm(n_embed)


    # TODO: fix this src,target mask right shifted outputs
    def forward(self, x):
        x = x + self.MHSAttention(x)
        x = self.layer_norm(x)
        x = self.block(x)
        return x

class Decoder(nn.Module):
    """
    Decoder for a transformer
    """

    def __init__(self, vocab_size, num_layers, n_embed, num_heads, block_size, dropout, device):
        self.word_embedding = nn.Embedding(vocab_size, n_embed)
        self.block = nn.ModuleList([DecoderBlock(n_embed, num_heads, block_size, dropout) for _ in num_layers])
        self.pos_encoder = PositionalEncoding(n_embed, block_size, device)

    # TODO: also fix output linear and softmax with target vocab
    def forward(self, x):
        x = self.word_embedding(x) + self.pos_encoder(x)
        for layer in self.block:
            x = layer(x)
        return x

