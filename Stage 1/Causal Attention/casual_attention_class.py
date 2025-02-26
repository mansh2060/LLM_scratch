import torch
import torch.nn as nn

inputs_1=torch.tensor(
    [[0.43,0.15,0.89], # your
     [0.55,0.87,0.66], # journey
     [0.57,0.85,0.64], # starts
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # one
     [0.05,0.80,0.55]  # step
     ],
    )
inputs_2 = torch.tensor(
      [[0.9755,0.6623,0.7943],
      [0.0189,0.4839,0.4241],
      [0.8391,0.4353,0.4203],
      [0.9441,0.5227,0.8878],
      [0.5871,0.5157,0.0359],
      [0.7861,0.7323,0.1976]] 
)
batch = torch.stack((inputs_1,inputs_2))
print(batch)
print(batch.shape)     # shape of the batch 

class CasualAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,wkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.w_query = nn.Linear(d_in,d_out,bias=wkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=wkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=wkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,inputs):
        batch_size,num_of_queries,embedding_dimension=inputs.shape
        keys = self.w_key(inputs)
        query = self.w_query(inputs)
        values = self.w_value(inputs)
        attention_score = query @ keys.transpose(1,2)
        attention_score.masked_fill(self.mask.bool()[:num_of_queries,:num_of_queries],-torch.inf)
        attention_weight = torch.softmax(attention_score/keys.shape[-1]**0.5,dim=-1)
        attention_weight=self.dropout(attention_weight)
        context_vector = attention_weight @ values
        return context_vector
    
torch.manual_seed(123)    
context_length = batch.shape[1]  
casual_attention = CasualAttention(d_in=3,d_out=2,context_length=context_length,dropout=0.0)
context_vector = casual_attention(batch)
print(context_vector.shape)
print(context_vector)