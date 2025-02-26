"""
here i need to mask all the non current elements i.e elements after the current element
for masking  i will create a lower triangular matrix using torch.trill 
"""


import torch.nn as nn
import torch
inputs=torch.tensor(
    [[0.43,0.15,0.89], # your
     [0.55,0.87,0.66], # journey
     [0.57,0.85,0.64], # starts
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # one
     [0.05,0.80,0.55]  # step
     ]
) 
class SelfAttention(nn.Module):
    def __init__(self, row_size, column_size,qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(row_size,column_size,bias=qkv_bias)
        self.w_key   = nn.Linear(row_size,column_size,bias=qkv_bias)
        self.w_value = nn.Linear(row_size,column_size,bias=qkv_bias)

    def forward(self,input_embeddings):
        query =  self.w_query(input_embeddings)
        key   =  self.w_key(input_embeddings)
        value =  self.w_value(input_embeddings)
        attention_score = query @ key.T
        key_dimension = key.shape[-1]
        attention_weight = torch.softmax(attention_score / key_dimension ** 0.5 , dim=-1)
        context_vector = attention_weight @ value
        return context_vector

torch.manual_seed(123)
self_attention=SelfAttention(3,2)
queries=self_attention.w_query(inputs)
keys=self_attention.w_key(inputs)
print(queries)
print(keys)

attention_scores = queries @ keys.T
print(f"attention_scores : {attention_scores}")

attention_weight = torch.softmax(attention_scores / keys.shape[-1]**0.5,dim=-1)
print(attention_weight)

context_length = attention_weight.shape[0]
mask_sample = torch.tril(torch.ones(context_length,context_length))
print(mask_sample)

mask_sample_1 = torch.tril(attention_weight)
print(mask_sample_1)

masked_matrix = attention_weight * mask_sample
print(masked_matrix)
