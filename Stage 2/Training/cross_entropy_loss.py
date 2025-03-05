import torch
import torch.nn as nn
import tiktoken
GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "embed_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}
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

class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.w_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",
                             torch.triu(torch.ones(context_length,context_length),diagonal=1)
                            )
        
        
    def forward(self,x):
        batch_size,num_tokens,d_in = x.shape
        
        keys = self.w_key(x)
        query = self.w_query(x)
        value = self.w_value(x)
        
        keys = keys.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        query = query.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        value = value.view(batch_size,num_tokens,self.num_heads,self.head_dim)
        
        keys = keys.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)
        
        attention_score =query @ keys.transpose(2,3)
        
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attention_score = attention_score.masked_fill(mask_bool , -torch.inf)
        
        attention_weight = torch.softmax(attention_score / keys.shape[-1] ** 0.5 , dim=-1)
        
        attention_weight = self.dropout(attention_weight)
        
        context_vector = (attention_weight @ value).transpose(1,2)
        context_vector = context_vector.contiguous().view(batch_size,num_tokens,self.d_out)
        context_vector = self.out_proj(context_vector)
        
        return context_vector
    
class GELUActivationFunction(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self,x):
        return  0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.0044715 * torch.pow(x,3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]),   # expansion
                                    GELUActivationFunction(),                            # activation function
                                    nn.Linear(4 * cfg["embed_dim"],cfg["embed_dim"]))    # contraction
        
    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["embed_dim"],
            d_out = cfg["embed_dim"],
            context_length= cfg["context_length"],
            dropout = cfg["drop_rate"],
            num_heads = cfg["n_heads"],
            qkv_bias = False
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNormalization(cfg["embed_dim"])
        self.norm2 = LayerNormalization(cfg["embed_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut = x
        norm1 = self.norm1(x)
        context_vector = self.att(norm1)
        dropped_context_vector =self.drop_shortcut(context_vector)
        output_1 = dropped_context_vector + shortcut
        shortcut = output_1
        norm2 = self.norm2(output_1)
        context_vector = self.att(norm2)
        dropped_context_vector = self.drop_shortcut(context_vector)
        output_2 = dropped_context_vector + shortcut
        return output_2
    


class GPTModel(nn.Module):
    def __init__(self,cfg ):
        super().__init__()
        self.token_embeddings = nn.Embedding(cfg["vocab_size"],cfg["embed_dim"])
        self.positional_embeddings = nn.Embedding(cfg["context_length"],cfg["embed_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range (cfg["n_layers"])])
        self.final_norm = LayerNormalization(cfg["embed_dim"])
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
    
model = GPTModel(GPT_CONFIG_124M)
inputs = torch.tensor(
    [[16833,3626,6100],
     [40,1107,588]])
target = torch.tensor(
    [[3626,6100,345],
     [107,588,11311]]
)
with torch.no_grad():
    logits=model(inputs)
probabilities = torch.softmax(logits,dim=-1)
print(probabilities)
print(probabilities.shape)
output = torch.argmax(probabilities,dim=-1,keepdim=True)
print(output)

tokenizer = tiktoken.get_encoding("gpt2")

def text_to_tokenids(text):
    token_ids = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    return token_ids

def tokenids_to_text(ids):
    ids = ids.squeeze(0)
    text = tokenizer.decode(ids.tolist())
    return text
target_text = tokenids_to_text(target[0])
predicted_text=tokenids_to_text(output[0].flatten())
print(target_text)
print(predicted_text)

batch_idx = 0
probabilities_1 = probabilities[batch_idx,[0,1,2],target[batch_idx]]
print("Probabilities 1 ",probabilities_1)

batch_idx = 1
probabilities_2 =probabilities[batch_idx,[0,1,2],target[batch_idx]]
print("Probabilities 2 ",probabilities_2)

log_probabs = torch.log(torch.cat((probabilities_1,probabilities_2)))
print(log_probabs)

log_probabs = log_probabs * -1
print(log_probabs)

print(log_probabs.mean())

flat_logits = logits.flatten(0,1)
flat_targets = target.flatten()

loss = torch.nn.functional.cross_entropy(flat_logits,flat_targets)
print(loss)

perplexity=torch.exp(loss)
print(perplexity)