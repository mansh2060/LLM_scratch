import tiktoken
tokenizer=tiktoken.get_encoding("gpt2")
text="This is Manish <|endoftext|> Hello! do you like some tea?"
ids=tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(ids)
text=tokenizer.decode(ids)
print(text)