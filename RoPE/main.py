import torch

class RoPE(torch.nn.Module):
    """RoPE Rotatory Positional Embedding for Transformers
       it is a technique used to encode positional information 
       using rotation matrices to encode both absolute and relative
       positional information.

    Args:
        d: int: the dimension of the input embeddings
        base: int: the base of the exponential term in positional encoding
    """

    def __init__(self,d,base):
        super().__init__()
        self.d = d
        self.base = base
        self.cos_cache = None
        self.sin_cache = None

    def build_cache(self,x):
        if self.cos_cache is not None and x.shape[0] <= self.cos_cache.shape[0]:
            return
        
        seq_length = x.shape[0]
        # theta = 10,000^(-2*i/d) or 1/10,000^(2i/d)
        theta = 1 / (self.base ** (torch.arange(0,self.d,2).float()/self.d)).to(x.device)
        # positional Index -> [0,1,2...seq-1]
        seq_idx = torch.arange(0,seq_length).float().to(x.device)
        # Calculates m*(theta) = [ [0*theta_1, 0...], [1*theta_1, theta_2...theta_d/2] ... [seq-1*(theta_1), seq-1*(theta_2)...] ]
        idx_theta = torch.einsum('i,j->ij',seq_idx,theta)
        # concate 2 theta matrics to generate ->  [theta_1, theta_2...theta_d]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        # Cache cos and sin values of theta
        self.cos_cache = idx_theta2.cos()[:, None, None, :]
        self.sin_cache = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self,x):
        d_2 = self.d//2
        # generate negative half of x
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) 

    def forward(self,x):
        self.build_cache(x)
        neg_half = self._neg_half(x)

        # form: [x_1cosTheta_1 - x_d/2sinTheta_d/2 ... ]
        x_rope = (x*self.cos_cache[:x.shape[0]]) + (neg_half*self.sin_cache[:x.shape[0]])
        return x_rope

# Testing the RoPE

RoPE_Module = RoPE(4,10000)
x = torch.randint(0,10, (3,4), dtype=torch.float)
x = x[:, None, None, :] # converts x [3,4] to 4d [3,1,1,4]
output = RoPE_Module(x)
print(output.shape)
