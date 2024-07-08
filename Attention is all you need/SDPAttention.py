import torch
import torch.nn as nn   

def SDPAttention(Q,K,V):
    """
    Scaled Dot Product Attention

    Args:
    Q: Queries matrix of shape (batch_size, num_queries, dk)
    K: Keys matrix of shape (batch_size, num_keys, dk)
    V: Values matrix of shape (batch_size, num_keys, dv)
    num_keys == num_queries

    Returns:
    Attention output of shape (batch_size, num_queries, dv)
    """
    dk = Q.shape[-1]
    # k -> (b,q,dk) -> (b,dk,q) , swaps last 2 dims
    K = K.transpose(-2,-1)
    # (b,q,dk) . (b,dk,q) -> (b,q,q) / dk**0.5
    scores = torch.matmul(Q,K) / torch.sqrt(torch.tensor(dk,dtype=torch.float32))
    scores_sm = torch.nn.functional.softmax(scores,dim=-1)
    attention_output = torch.matmul(scores_sm,V)
    return attention_output

batch_size = 2
num_queries = 3
num_keys = 4
dk = 5
dv = 6

Q = torch.randn(batch_size, num_queries, dk).to('cuda')
K = torch.randn(batch_size, num_keys, dk).to('cuda')
V = torch.randn(batch_size, num_keys, dv).to('cuda')

output = SDPAttention(Q, K, V)
print(output)