import torch
import torch.nn as nn   

class Head(nn.Module):
    """
    Head of Self Attention
    """
    def __init__(self, n_head,n_embed,block_size,dropout):
        super().__init__()
        self.key = nn.Linear(n_embed,n_head,bias=False)
        self.queries = nn.Linear(n_embed,n_head,bias=False)
        self.value = nn.Linear(n_embed,n_head,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular identity matrix
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        b,t,c = x.shape
        k = self.key(x)
        q = self.queries(x)
        v = self.value(x)
        # k -> (b,q,dk) -> (b,dk,q) , swaps last 2 dims
        k = k.transpose(-2,-1)
        # (b,q,dk) . (b,dk,q) -> (b,q,q) / c**0.5
        scores = torch.matmul(q,k) / torch.sqrt(torch.tensor(c,dtype=torch.float32))
        # makes a mask of shape (t,t) of T/F and replaces T with -inf for sm->0 (e**-inf=0, no information from future)
        scores = scores.masked_fill(self.tril[:t, :t] == 0, float('-inf')) # (b,t,t)
        scores_sm = torch.nn.functional.softmax(scores,dim=-1)
        attention_output = torch.matmul(scores_sm,v)
        return attention_output

class MHSAttention(nn.Module):
    """
    Multi Head Self attention
    """

    def __init__(self, num_heads,n_embed,block_size,dropout):
        super().__init__()
        self.head_size = n_embed // num_heads
        self.heads = nn.ModuleList([Head(self.head_size,n_embed,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out




dropout = 0.0
num_heads = 4
block_size = 128 # context length, B
num_queries = 100 # number of queries, T
n_embed = 128 # embedding size, C
head_size = n_embed // num_heads


x = torch.randn(block_size, num_queries, n_embed)
MHSAttentionBlock = MHSAttention(num_heads,n_embed,block_size,dropout)
print(MHSAttentionBlock(x))