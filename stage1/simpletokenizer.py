from text_reader import vocab
import re
class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int=vocab
        self.int_to_str={idx:tokens for tokens,idx in vocab.items()}

    def encoder(self,text):
        tokens=re.split(r'([,.:;?_!"()\']|--|\s)',text)
        processed_tokens=[word.strip() for word in tokens if word.strip()]
        processed_tokens=[elements if elements in self.str_to_int else "|<unk>|" for elements in processed_tokens]
        ids=[self.str_to_int[items] for items in processed_tokens]
        return ids

    def decoder(self,ids):
        text=" ".join([self.int_to_str[idx] for idx in ids])
        processed_text=re.sub(r'\s+([,.?!"()\'])', r'\1',text)
        return processed_text

tokenizer=SimpleTokenizerV1(vocab)
text="Hello! Well!--even through the prism of Hermia's tears I felt able  to face the fact with equanimity. "
ids=tokenizer.encoder(text)
text=tokenizer.decoder(ids)
print(ids)
print(text)