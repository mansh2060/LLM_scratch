import torch
import torch.nn as nn
inputs = torch.tensor(
        [[-0.1115,  0.1204, -0.3696, -0.2404, -1.1969],
        [ 0.2093, -0.9724, -0.7550,  0.3239, -0.1085]])
class LayerNormalization(nn.Module):
    def __init__(self,embedding_dimension):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dimension))
        self.shift = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        variance = x.var(dim = -1,keepdim = True,unbiased = False)
        normalize_x = (x - mean ) / torch.sqrt(variance + self.eps)
        return self.scale * normalize_x + self.shift

torch.manual_seed(123)   
layer_normalization = LayerNormalization(embedding_dimension=5)
normalized_output = layer_normalization(inputs)  
print(normalized_output)
mean = normalized_output.mean(dim=-1,keepdim = True)
variance = normalized_output.var(dim = -1 ,keepdim = True)
print(mean)
print(variance)