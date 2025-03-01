import torch
import torch.nn as nn
import tiktoken
"""
token _ ids  -->  token  embedding generate    vocab_size * embedding_dimension
positional_embedding    context_length * embedding_dimension
drop_embedding ---->  dropout()
"""
GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "embed_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

class DummyGPTModel(nn.Module):
    def __init__(self,cfg ):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg["vocab_size"],cfg["embed_dim"])
        self.positional_embeddings = nn.Embedding(cfg["context_length"],cfg["embed_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range (cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"],cfg["vocab_size"],bias=False)

    def forward(self,x):
        batch_size,sequence_length = x.shape
        token_embeddings = self.token_embeddings(x)
        positional_embeddings = self.positional_embeddings(torch.arange(sequence_length, device=x.device))
        input_embeddings = token_embeddings + positional_embeddings
        input_embeddings = self.drop_embedding(input_embeddings)
        input_embeddings = self.transformer_blocks(input_embeddings)
        input_embeddings = self.final_norm(input_embeddings)
        logits = self.out_head(input_embeddings)
        return logits
    

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # The parameters here are just to mimic the LayerNorm interface.

    def forward(self, x):
        # This layer does nothing and just returns its input.
        return x
    
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every efforts move you"
text2 = "Every day holds a"
token_ids_1 = tokenizer.encode(text1)
token_ids_2 = tokenizer.encode(text2)
batch.append(torch.tensor(token_ids_1))
batch.append(torch.tensor(token_ids_2))
batch = torch.stack(batch,dim=0)
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Outputs shape",logits.shape)
print(logits)