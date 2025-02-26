"""
attention weight = normalization of attention score
d_k  = scaling by root under dimension key
attention score / d_k
softmax (normalize across columns keeping row same dim=-1)
"""
import torch
import math
attention_score=torch.tensor(
                [[0.9231, 1.3545, 1.3241, 0.7910, 0.4032, 1.1330],
                [1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440],
                [1.2544, 1.8284, 1.7877, 1.0654, 0.5508, 1.5238],
                [0.6973, 1.0167, 0.9941, 0.5925, 0.3061, 0.8475],
                [0.6114, 0.8819, 0.8626, 0.5121, 0.2707, 0.7307],
                [0.8995, 1.3165, 1.2871, 0.7682, 0.3937, 1.0996]]
                )
key_vector=torch.tensor(
            [[0.3669, 0.7646],
            [0.4433, 1.1419],
            [0.4361, 1.1156],
            [0.2408, 0.6706],
            [0.1827, 0.3292],
            [0.3275, 0.9642]]
            )
key_dimension=key_vector.shape[-1]
# attention weight for second attention score
attention_score_2=attention_score[1]
print(attention_score_2)
print(key_dimension)
scaling_score_2=attention_score_2/math.sqrt(key_dimension)
print(scaling_score_2)
attention_weight_2=torch.softmax(scaling_score_2,dim=0)
print(attention_weight_2)

# attention score for all queries
scaling_score=attention_score / math.sqrt(key_dimension)
print(scaling_score)
attention_weight=torch.softmax(scaling_score,dim=-1)
print(attention_weight)