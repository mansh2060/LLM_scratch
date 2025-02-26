## token embeddings matrix  for LLM  + weight embedding matrix 
import torch
import torch.nn as nn
'''
creating a matrix with the row size of no of tokens ids and column size of embedding_dimension
basically  vocab_size * embedding_dimension

'''
input_ids=torch.tensor([2,3,5,1])
vocab_size=6
output_dim=3
torch.manual_seed(123)  # matrix create --> randomly genearted value 
embedding_layer=torch.nn.Embedding(vocab_size,output_dim)  # this is used for mapping a token id to its vectors
print(embedding_layer)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids)) 