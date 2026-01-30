import torch
import torch.nn as nn
import tiktoken
from Attention import  MultiHeadAttention # pyright: ignore[reportMissingImports]

# layer normalization part 
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # calculate the mean for each row 
        var = x.var(dim=-1, keepdim=True, unbiased=False) #calulate the variance of each row , unbiased=False >> for calc  population  devide by n 
        norm_x = (x - mean) / torch.sqrt(var + self.eps) #The variable eps is added to the variance to prevent division by zero during normalization
        return self.scale * norm_x + self.shift 

# the GELU activation function block 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
# feed forward part
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), #layer1 >input 768 dim and output 4*768 like gpt2
            GELU(), #the activation function between them 
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), #layer1 >input 4*768 dim and output 768 like gpt2
        )

    def forward(self, x):
        return self.layers(x)
# transformers part        
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            # cfg in the first line of the page the gpt-2 parameters
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.ff = FeedForward(cfg) # the feed forward after attention 
        self.norm1 = LayerNorm(cfg["emb_dim"])# normalization before attention 
        self.norm2 = LayerNorm(cfg["emb_dim"]) # normalization after feed forward  

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x # >> the inputs 
        x = self.norm1(x) #1 the normalization layer 
        # multi head attention 
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size] 
        x = self.drop_shortcut(x) #the dropout 
        x = x + shortcut  # Add the original input back x = data_after_attention + input embedings

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x