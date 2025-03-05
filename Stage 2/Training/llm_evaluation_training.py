import os
import urllib.request
import tiktoken
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

print(text_data[:99])
print(text_data[-99:])

total_tokens= len(text_data)
print(total_tokens)


GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 256,
    "embed_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

tokenizer=tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(text_data)
print(len(token_ids))

train_ratio = 0.90
split_index = int(train_ratio * len(text_data))
training_data = text_data[:split_index]
validation_data = text_data[split_index:]

class GPTDataset(Dataset):
    def __init__(self,text,context_length,stride,tokenizer):
        super().__init__()
        self.input_index = []
        self.output_index = []
        token_ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids)-context_length,stride):
            self.input_chunk = token_ids[i:context_length+i]
            self.output_chunk = token_ids[i+1:context_length+i+1]
            self.input_index.append(torch.tensor(self.input_chunk))
            self.output_index.append(torch.tensor(self.output_chunk))

    def __len__(self):
        return len(self.input_index)
    
    def __getitem__(self, index):
        return self.input_index[index],self.output_index[index]
    
def create_data_loader(text,batch_size=9,context_length=256,stride=256,shuffle=True,drop_last=True,num_workers=0):
    gpt_dataset=GPTDataset(text,context_length,stride,tokenizer)
    dataloader = DataLoader(gpt_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0
                            )
    return dataloader

torch.manual_seed(123)

train_loader = create_data_loader(training_data,
                                  batch_size=2,
                                  context_length=GPT_CONFIG_124M["context_length"],
                                  stride=GPT_CONFIG_124M["context_length"],
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=0
                                  )

test_loader = create_data_loader(validation_data,
                                 batch_size=2,
                                 context_length =GPT_CONFIG_124M["context_length"],
                                 stride=GPT_CONFIG_124M["context_length"],
                                 drop_last=False,
                                 shuffle=False,
                                 num_workers=0)

if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
    print("Not Enough Tokens to Train The Dataset")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not Enough Tokens To Validate The Dataset")

print("Train_loader Shape")
for x, y in train_loader:
    print(x.shape,y.shape)
print(len(train_loader))

print("Test_loader shape")
for x,y in test_loader:
    print(x.shape,y.shape)
print(len(test_loader))

train_tokens = 0
for input_batch, _ in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, _ in test_loader:
    val_tokens += input_batch.numel()

print("Train Tokens         :",train_tokens)
print("Validation Tokens    :",val_tokens)
print("Total Tokens         :",train_tokens + val_tokens)

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
    
def calculate_loss_batch(input_batch,target_batch,model,device):
        input_batch,target_batch = input_batch.to(device) , target_batch.to(device)
        model = GPTModel(GPT_CONFIG_124M)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
        return loss

def calculate_loss_data_loader(data_loader,model,device,num_batches=None):
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(len(data_loader),num_batches)
        for i,(input_batch,target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calculate_loss_batch(input_batch,target_batch,model,device)
                total_loss += loss.item()
            else:
                break
        return total_loss /num_batches
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 


torch.manual_seed(123) 
with torch.no_grad(): 
    train_loss = calculate_loss_data_loader(train_loader, model, device)
    val_loss = calculate_loss_data_loader(test_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)