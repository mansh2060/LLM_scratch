from torch.utils.data import Dataset,DataLoader
import tiktoken
import torch
tokenizer=tiktoken.get_encoding("gpt2")
class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids=[]
        self.output_ids=[]

        token_ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk=token_ids[i:max_length+i]               #chunk of tokens break input 
            output_chunk=token_ids[i+1:max_length+i+1]          #chunk of tokens break output
            self.input_ids.append(torch.tensor(input_chunk))    
            self.output_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index],self.output_ids[index]

def create_data_loader(text,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    dataset=GPTDataset(text,tokenizer,max_length,stride)
    dataloader=DataLoader(dataset,
               batch_size=batch_size,
               shuffle=shuffle,
               drop_last=True,
               num_workers=0)
    return dataloader


with open("extracted_data.txt",'r',encoding='utf-8') as file:
    text=file.read()

data_loader=create_data_loader(text,batch_size=1,max_length=12,stride=4,shuffle=False)

data_iter=iter(data_loader)
first_batch=next(data_iter)
second_batch=next(data_iter)
third_batch=next(data_iter)
print(first_batch)
print(second_batch)
print(third_batch)