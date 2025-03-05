'''
in this programme first i am making embedding_layer_matrix which is of size (vocab_size,embedding_dim)
positional_embedding_layer which of size (batch_size,max_length,embedding_size)
Key Terminologies
Batch_size 
max_length(sequence_length,context_length) 
embedding_dimension 
vocab_size 
stride 
'''
from GPT_Dataset_Dataloader import create_data_loader
import torch
with open("The_Verdict.txt",'r',encoding='utf-8') as file:
    text=file.read()

vocab_size=50257
embedding_dim=768
context_length=8
token_embedding_layer=torch.nn.Embedding(vocab_size,embedding_dim)
positional_embedding_layer=torch.nn.Embedding(context_length,embedding_dim)

data_loader=create_data_loader(text,batch_size=4,max_length=8,stride=4,shuffle=False)
data_iter=iter(data_loader)
first_batch=next(data_iter)
inputs,target=first_batch
print(f"Inputs : {inputs}")
print(f"Target : {target}")
print(inputs.shape)       # vocab size * embedding_dimension (For GPT-2 this is 50257 * 768)
token_embeddings=token_embedding_layer(inputs)
print(token_embeddings)
print(token_embeddings.shape)  # batch_size * max_length * embedding_dimension : 4 * 8 * 256  


positional_embedding=positional_embedding_layer(torch.arange(context_length))
print(positional_embedding.shape)

input_embedding=token_embeddings + positional_embedding
print(input_embedding.shape)