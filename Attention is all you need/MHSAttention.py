import torch
import torch.nn as nn   


class MHSAttention(nn.Module):
  """
  Multi Head Self attention of a Transformer
  """

  def __init__(self,n_embed,num_heads):
    super().__init__()
    self.n_embed = n_embed
    self.num_heads = num_heads
    self.head_dim = self.n_embed // self.num_heads

    assert (self.head_dim*self.num_heads == self.n_embed), f"{self.head_dim} * {self.num_heads} != {self.n_embed}"
    self.key = nn.Linear(self.n_embed,self.n_embed)
    self.query = nn.Linear(self.n_embed,self.n_embed)
    self.value = nn.Linear(self.n_embed,self.n_embed)
    
    self.sf = nn.Softmax(dim=3)
    self.fc = nn.Linear(self.head_dim * num_heads,n_embed)

  def forward(self, key, query, value,mask = None):
    key = self.key(key)
    query = self.query(query)
    value = self.value(value)
    # key,query,value shape: [B,T,C]

    B = key.shape[0]
    key_len,query_len,value_len = key.shape[1], query.shape[1], value.shape[1]
    
    # reshape ensures the shape might change due to the decoder block attention
    # which was calculated using query of decoder but keys,values of encoder's output
    key = key.reshape(B,key_len,self.num_heads,self.head_dim)
    query = query.reshape(B,query_len,self.num_heads,self.head_dim)
    value = value.reshape(B,value_len,self.num_heads,self.head_dim)

    # converts q[nqhd], k[nkhd]
    # and dot product of query and key along last dim.
    energies = torch.einsum("nqhd,nkhd->nhqk",[query,key])

    if mask is not None:
      # e**-inf->0 , no information from future
      energies = energies.masked_fill(mask==0,float("-inf"))

    attention = self.sf(energies/(self.n_embed)**0.5)
    attention_score = torch.einsum('nhql,nlhd->nqhd',[attention,value])
    # concat all the heads
    attention_score = attention_score.reshape(B,query_len,self.n_embed)
    attention_score = self.fc(attention_score)
    return attention_score

# Testing the MHSAttention

# n_embed = 128
# num_heads = 4
# key = torch.randn(128,100,128)
# query = torch.randn(128,100,128)
# value = torch.randn(128,100,128)
# mask = torch.ones(100,100)

# MHSAttentionBlock = MHSAttention(n_embed,num_heads)
# print(MHSAttentionBlock(key,query,value,mask).shape, MHSAttentionBlock(key,query,value,mask).isnan().all())