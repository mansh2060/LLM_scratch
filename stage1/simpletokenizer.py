from text_reader import extracted_data
import re
#variable initialize , encoder(return ids) , decoder(return text)

class SimpleTokenizerV1:
        def __init__(self,extracted_data):
            self.extracted_data=extracted_data

        def encoder(self):
            text=re.split(r'([,.?;_:"\']|--|\s)',self.extracted_data)
            text=[word.strip() for word in text if word.strip()] 
            vocab={tokens: idx for idx,tokens in enumerate(text)} # dictionary key-value
            text=[word if word in vocab else '<|unk|>' for word in vocab ]
            vocab={tokens: idx for idx,tokens in enumerate(text)} 
            ids=[vocab[words] for words in vocab]
            return vocab,ids
        
        def decoder(self,ids):
            vocab,_=self.encoder()
            decoded_tokens=[]
            index_to_token={idx:tokens for tokens,idx in vocab.items()}
            for index in ids:
                if index in index_to_token:
                    decoded_tokens.append(index_to_token[index])
            text=" ".join(decoded_tokens)
            text=re.sub(r'\s+([,.?";:!()\'])',r'\1',text)
            return text

simpletokens=SimpleTokenizerV1(extracted_data)
_,ids=simpletokens.encoder()
print(ids)
text=simpletokens.decoder(ids)  
print(text)
 

            
         

