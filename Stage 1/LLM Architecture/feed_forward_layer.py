import torch.nn as nn
import torch
GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "embed_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}
class GELUActivationFunction(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self,x):
        return  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.0044715 * torch.pow(x,3))))
    
"""
input layer (768 )   ---- > hidden_layer 1 (4 * 768)
Activation function  (Non Linear + dead state or alive)
hidden_layer_1 (4*768) ----> output_layer (768)

"""
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),   # expansion
                                    GELUActivationFunction(),                            # activation function
                                    nn.Linear(4 * cfg["embed_dim"],cfg["embed_dim"]))    # contraction
        
    def forward(self,x):
        return self.layers(x)

torch.manual_seed(123)
batch = torch.randn(2,3,768)
feed_forward_layer = FeedForward(GPT_CONFIG_124M)
trained_output = feed_forward_layer(batch)
print(trained_output)
print(trained_output.shape)
