import tiktoken
from pdfminer.high_level import extract_text
extracted_data=extract_text('The_Verdict.pdf')

filename='extracted_data.txt'
with open(filename,'w',encoding='utf-8') as file:
    file.write(extracted_data)

with open(filename,'r',encoding='utf-8') as file:
    data=file.read()
tokenizer=tiktoken.get_encoding("gpt2")
ids=tokenizer.encode(data)
ids=ids[1500:]
text=tokenizer.decode(ids)
context_size=4
for i in range(1,context_size+1):
    input_=ids[:i]
    output_=ids[i]
    print(f"{input_}-->{output_}")
for i in range(1,context_size+1):
    input_=ids[:i]
    output_=ids[i]
    print(f"{tokenizer.decode(input_)} --> {tokenizer.decode([output_])}")

