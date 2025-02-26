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
'''
input_embeddings  * weight matrix (3 * 2)       = query_matrix
input_embeddings  * key matrix (3 * 2)          = key_matrix
input_embeddings  * value_matrix (3 * 2)        = value_matrix
'''

print(inputs.shape)     # size of matrix
print(inputs.shape[0])  # it represents row size
print(inputs.shape[1])  # it represents column size

row_size=inputs.shape[1]
column_size=2

torch.manual_seed(123)
w_query=torch.nn.Parameter(torch.rand(row_size,column_size),requires_grad=False)
w_key=torch.nn.Parameter(torch.rand(row_size,column_size),requires_grad=False)
w_value=torch.nn.Parameter(torch.rand(row_size,column_size),requires_grad=False)

print(w_query)
#print(w_key)
#print(w_value)

# calculation of key , query , value for one input 
inputs_1=inputs[1]
query_vector = inputs_1 @ w_query
key_vector   = inputs_1 @ w_key
value_vector = inputs_1 @ w_value

print(query_vector)
#print(key_vector)
#print(value_vector)

# calculation of key , query , value for all inputs

query_vector = inputs @ w_query
key_vector   = inputs @ w_key
value_vector = inputs @ w_value

#print(query_vector)
#print(key_vector)
#print(value_vector)