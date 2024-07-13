import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer block
    """

    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        source_pad_idx,
        target_pad_idx,
        n_embed,
        num_layers,
        num_heads,
        dropout,
        device,
        max_length,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.src_pad_idx = source_pad_idx
        self.trg_pad_idx = target_pad_idx
        self.device = device
        self.head_dim = n_embed // num_heads
        self.encoder = Encoder(
            source_vocab_size,
            num_layers,
            n_embed,
            num_heads,
            max_length,
            dropout,
            device,
        )
        self.decoder = Decoder(
            target_vocab_size,
            num_layers,
            n_embed,
            num_heads,
            max_length,
            dropout,
            device,
        )

    def generate_source_mask(self, source):
        src_mask = (source != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def generate_target_mask(self, target):
        N, trg_len = target.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, source, target):
        source_mask = self.generate_source_mask(source)
        target_mask = self.generate_target_mask(target)
        source_encoding = self.encoder(source, source_mask)
        out = self.decoder(target, source_encoding, source_mask, target_mask)
        return out
