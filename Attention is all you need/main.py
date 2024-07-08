import torch
import torch.nn as nn
from Transformer import Transformer

# TODO: add variables here

model = Transformer() # also add the parameters
model = model.to('cuda')

# TODO: also add training loop and dataset preprocessing