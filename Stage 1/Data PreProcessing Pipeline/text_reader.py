import re
from pdfminer.high_level import extract_text
extracted_data=extract_text('The_Verdict.pdf')

#creating a vocabulary of extracted text
token_list=re.split(r'([,.:;?_!"()\']|--|\s)',extracted_data)
print(f"Length of all_words   :{len(token_list)}")

token_list=[word.strip() for word in token_list if word.strip()]
print(f"Length of stripped_words   :{len(token_list)}")

token_list=sorted(set(token_list))
print(f"Length of sorted_words   :{len(token_list)}")

token_list.extend(["|<unk>|","|<endoftext>|"])
print(f"Length of Vocabulary   :{len(token_list)}")

vocab={word:idx for idx,word in enumerate(token_list)}

for tokens,idx in (list(vocab.items())[-5:]):
    print(f"{tokens}:{idx}")