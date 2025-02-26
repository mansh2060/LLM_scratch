''''
in this code i am getting attention weight using upper triangular matrix using torch.triu in order to 
prevent the data leakage 
first -->  upper triangular matrix is created  diagonal = 1
second --> 1 is filled with -inf and remaining with attention score
'''
import torch
attention_mask=torch.tensor(
        [[0.1717, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1636, 0.1749, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1637, 0.1749, 0.1746, 0.0000, 0.0000, 0.0000],
        [0.1636, 0.1704, 0.1702, 0.1652, 0.0000, 0.0000],
        [0.1667, 0.1722, 0.1721, 0.1618, 0.1633, 0.0000],
        [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]])
 
attention_weight=attention_mask/attention_mask.sum(dim=1,keepdim=True)
print(attention_weight)

attention_score=torch.tensor(
        [[0.3111, 0.3479, 0.3471, 0.1714, 0.2350, 0.1928],
        [0.1655, 0.2602, 0.2576, 0.1445, 0.1384, 0.1790],
        [0.1667, 0.2602, 0.2577, 0.1443, 0.1391, 0.1784],
        [0.0510, 0.1080, 0.1064, 0.0643, 0.0476, 0.0835],
        [0.1415, 0.1875, 0.1863, 0.0987, 0.1121, 0.1174],
        [0.0476, 0.1192, 0.1171, 0.0731, 0.0477, 0.0966]],
       )

keys=torch.tensor(
        [[0.3669, 0.7646],
        [0.4433, 1.1419],
        [0.4361, 1.1156],
        [0.2408, 0.6706],
        [0.1827, 0.3292],
        [0.3275, 0.9642]])
context_length = attention_score.shape[0]
upper_masked_matrix=torch.triu(torch.ones(context_length,context_length),diagonal=1)   
print(upper_masked_matrix)   

upper_masked_matrix=upper_masked_matrix.masked_fill(upper_masked_matrix.bool(),-torch.inf)
print(upper_masked_matrix)

attention_score=attention_score.masked_fill(upper_masked_matrix.bool(),-torch.inf)
print(attention_score)

attention_weight=torch.softmax(attention_score/keys.shape[-1]**0.5,dim=-1)
print(attention_weight)