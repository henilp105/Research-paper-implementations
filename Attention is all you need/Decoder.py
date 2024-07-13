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
        super().__init__()
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

    def __init__(self, target_vocab_size, num_layers, n_embed, num_heads, block_size, dropout, device):
        super().__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, n_embed)
        self.block = nn.ModuleList([DecoderBlock(n_embed, num_heads, block_size, dropout) for _ in range(num_layers)])
        self.pos_encoder = PositionalEncoding(n_embed, block_size, device)
        self.fc_out = nn.Linear(n_embed, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, source_mask,target_mask):
        x = self.dropout(self.word_embedding(x) + self.pos_encoder(x))

        for layer in self.block:
            x = layer(x,enc_out,source_mask,target_mask)
        x = self.fc_out(x)
        return x

