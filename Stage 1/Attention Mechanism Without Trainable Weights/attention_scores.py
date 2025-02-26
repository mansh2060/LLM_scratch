# your journey starts with one step
'''
here i am finding the attention score of each word with respect to another word
'''
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
attention_score_matrix=torch.empty(inputs.shape[0])
#print(attention_score_matrix.shape)
query=inputs[1]
for index,embedding in enumerate(inputs):
    attention_score_matrix[index]=torch.dot(embedding,query)
print(attention_score_matrix)

attention_score_matrix_normalize=attention_score_matrix/attention_score_matrix.sum() # manual normalization 
#print(f"Attention Weights : {attention_score_matrix_normalize}")
#print(f"Sum               : {attention_score_matrix_normalize.sum()}")

attention_scores=inputs @ inputs.T # here i am performing multiplication opeartion between inputs * transpose of inputs using Linear Algebra
print(attention_scores)

attention_weight=torch.softmax(attention_scores,dim=-1)
print(attention_weight)

