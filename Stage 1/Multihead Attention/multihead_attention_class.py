import torch.nn as nn
import torch
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length,context_length),diagonal=1)
                            )
        
        
    def forward(self,x):
        batch_size,num_tokens,d_in = x.shape
        
        keys = self.w_key(x)
        query = self.w_query(x)
        value = self.w_value(x)
        
        keys = keys.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        query = query.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        value = value.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        
        keys = keys.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)
        
        attention_score =query @ keys.transpose(2,3)
        
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attention_score = attention_score.masked_fill(mask_bool , -torch.inf)
        
        attention_weight = torch.softmax(attention_score / keys.shape[-1] ** 0.5 , dim=-1)
        
        attention_weight = self.dropout(attention_weight)
        
        context_vector = (attention_weight @ value).transpose(1,2)
        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.d_out)
        context_vector = self.out_proj(context_vector)
        
        return context_vector

torch.manual_seed(123)
inputs =torch.tensor(
        [[0.3162, 0.7007, 0.8276, 0.4542, 0.4278, 0.9586],
        [0.1797, 0.4201, 0.8284, 0.7746, 0.6332, 0.4154],
        [0.3236, 0.7943, 0.5929, 0.6950, 0.4438, 0.5973]]
        )
batch = torch.stack((inputs,inputs),dim=0)
print(batch.shape)
batch_size,num_tokens,d_in = batch.shape
d_out = 6
context_length = num_tokens
multi_head_attention=MultiHeadAttention(d_in,d_out,context_length,0.0,num_heads=2)
context_vector = multi_head_attention(batch)
print(context_vector)
print(context_vector.shape)