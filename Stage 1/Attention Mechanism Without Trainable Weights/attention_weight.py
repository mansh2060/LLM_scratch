import torch
from attention_scores import attention_score_matrix,attention_scores
attention_scores_1=attention_score_matrix
attention_scores_all=attention_scores
#matrix=torch.tensor([1,0.00001,500,2,-4])
def softmax_normalize_with_naive(attention_score_matrix_normalize):
    return torch.exp(attention_score_matrix_normalize)/torch.exp(attention_score_matrix_normalize).sum(dim=0)

normalized_softmax=softmax_normalize_with_naive(attention_scores_1)
sum_=normalized_softmax.sum()
print(normalized_softmax)
print(sum_)

def softmax_normalize_without_naive(attention_score_matrix):
    return torch.exp((attention_score_matrix)-max(attention_score_matrix))/torch.exp((attention_score_matrix)-max(attention_score_matrix)).sum(dim=0)

normalized_softmax=softmax_normalize_without_naive(attention_scores_1)
sum_=normalized_softmax.sum()
print(normalized_softmax)
print(sum_)

def pytorch_normalization(attention_score_matirx):
    return torch.softmax(attention_score_matirx,dim=0)  # softmax --> sotmax_normalization_without_naive

def pytorch_normalization_(attention_score_matirx):
    return torch.softmax(attention_score_matirx,dim=-1)
normalized_output=pytorch_normalization(attention_scores_1)
attention_weight=pytorch_normalization_(attention_scores_all)
print(normalized_output)
print(normalized_output.sum())
print(attention_weight)
print(attention_weight.sum())

