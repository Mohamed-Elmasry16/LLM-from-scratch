import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
       # intial values 
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        # intialize the weight matrices 
        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)

        # to applay the mask of causal attention
        self.register_buffer("mask",torch.triu(torch.ones(context_length, context_length),diagonal=1))
        # to applay dropout 
        self.dropout = nn.Dropout(dropout)

        # this is like a mixer to mix the weights from the heads together 
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
    def forward(self,x):
        b, num_tokens, d_in = x.shape

        keys = self.W_K(x) # Shape: (batch, num_tokens, d_out)
        queries = self.W_Q(x)
        values = self.W_V(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # calculate the attention weights 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # dropout to enhance training 
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim) >> to get d_out 
        context_vec = (attn_weights @ values).transpose(1, 2) 

        # Combine heads, where self.d_out = self.num_heads * self.head_dim >> final context vector for each word 
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # the mixer will mex the result of each head with other to consider all relations that each head collect
        context_vec = self.out_proj(context_vec) 


        return context_vec






