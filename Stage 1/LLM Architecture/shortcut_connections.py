import torch.nn as nn
import torch
"""
input_layer ---> hidden_layer 1       , 3
hidden_layer 1 ----> hidden_layer 2   , 3 
hidden_layer 2 -----> hidden_layer 3  , 3
hidden_layer 3 ------> hidden_layer 4 , 3
hidden_layer 4 ------> hidden_layer 5 , 3
hidden_layer 5 ------> output_layer   , 1
"""
class GELUActivationFunction(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self,x):
        return  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.0044715 * torch.pow(x,3))))
    
class ShortCutConnection(nn.Module):
    def __init__(self,layers_size,use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(layers_size[0],layers_size[1]),GELUActivationFunction()),
                                    nn.Sequential(nn.Linear(layers_size[1],layers_size[2]),GELUActivationFunction()),
                                    nn.Sequential(nn.Linear(layers_size[2],layers_size[3]),GELUActivationFunction()),
                                    nn.Sequential(nn.Linear(layers_size[3],layers_size[4]),GELUActivationFunction()),
                                    nn.Sequential(nn.Linear(layers_size[4],layers_size[5]),GELUActivationFunction())])
        
    def forward(self,x):
        for layer in self.layers:
            layer_output = layer(x)  
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x
               
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) # specify random seed for the initial weights for reproducibility
model_without_shortcut = ShortCutConnection(
layer_sizes, use_shortcut=False
)

def print_gradients(model,x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output,target)
    loss.backward()       # dloss / dweight
    for name,param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

torch.manual_seed(123)
model_with_shortcut = ShortCutConnection(
layer_sizes, use_shortcut=True
)

print_gradients(model_without_shortcut, sample_input)
print_gradients(model_with_shortcut,sample_input)