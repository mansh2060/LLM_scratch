import torch.nn as nn
import torch
class SelfAttention(nn.Module):
    def __init__(self, row_size, column_size):
        super().__init__()
        self.w_query = nn.Parameter(torch.rand(row_size,column_size))
        self.w_key   = nn.Parameter(torch.rand(row_size,column_size))
        self.w_value = nn.Parameter(torch.rand(row_size,column_size))

    def forward(self,input_embeddings):
        query = input_embeddings @ self.w_query
        key   = input_embeddings @ self.w_key
        value = input_embeddings @ self.w_value
        attention_score = query @ key.T
        key_dimension = key.shape[-1]
        attention_weight = torch.softmax(attention_score / key_dimension ** 0.5 , dim=-1)
        context_vector = attention_weight @ value
        return context_vector

inputs=torch.tensor(
    [[0.43,0.15,0.89], # your
     [0.55,0.87,0.66], # journey
     [0.57,0.85,0.64], # starts
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # one
     [0.05,0.80,0.55]  # step
     ]
)
torch.manual_seed(123)
self_attention_mechanism = SelfAttention(3,2)
output = self_attention_mechanism(inputs)
print(output)