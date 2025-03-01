import torch
import torch.nn as nn
"""
GELU(x) = 0.5x(1 + tanh[sqrt(2/pi) (x+0.004715 x^3)])
"""
class GELUActivationFunction(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self,x):
        return  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.0044715 * torch.pow(x,3))))
    