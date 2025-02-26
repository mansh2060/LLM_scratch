import torch
from attention_scores import attention_score_matrix,attention_scores
from attention_weight import pytorch_normalization,pytorch_normalization_
"""
context_vector = 0
context_score  = (embedding_vector * attention_weight)
context_vector = context_score + context_vector
"""
embedding_vector=torch.tensor(
    [[0.43,0.15,0.89], # your
     [0.55,0.87,0.66], # journey
     [0.57,0.85,0.64], # starts
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # one
     [0.05,0.80,0.55]  # step
     ]
)
matrix=attention_score_matrix
query=embedding_vector[1]
attention_weight=pytorch_normalization(matrix)
context_vector=torch.zeros(query.shape)
for index,element in enumerate(embedding_vector):
    context_score  = element * attention_weight[index]
    context_vector = context_vector + context_score
print(context_vector)
attention_scores_all=attention_scores
attention_weight=pytorch_normalization_(attention_scores)
context_vector_all=attention_weight @ embedding_vector
print(context_vector_all)

