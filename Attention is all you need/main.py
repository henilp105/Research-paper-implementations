import torch
import torch.nn as nn
from Transformer import Transformer


source_vocab_size = 100
target_vocab_size = 100
source_pad_idx = 0
target_pad_idx = 0
n_embed = 512
num_layers = 6
num_heads = 8
dropout = 0.1
device = 'cpu'
max_length = 100
seq_len = 10
batch_size = 16

model = Transformer(source_vocab_size,
        target_vocab_size,
        source_pad_idx,
        target_pad_idx,
        n_embed,
        num_layers,
        num_heads,
        dropout,
        device,
        max_length)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inputs = torch.randint(low=0, high=10, size=(batch_size, seq_len)).to(device)
target = torch.randint(low=0, high=10, size=(batch_size, seq_len+3)).to(device)

preds = model(inputs, target)

print(preds.shape,preds.isnan().all())
