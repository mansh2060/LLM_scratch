import torch
torch.manual_seed(123)
matrix=torch.ones(6,6)
print(matrix)
upper_masked_matrix=torch.triu(torch.rand(6,6),diagonal=1)
print(upper_masked_matrix)
dropout=torch.nn.Dropout(0.5)
dropout_matrix_1=dropout(matrix)
dropout_matrix_2=dropout(upper_masked_matrix)
print(dropout_matrix_1)
print(dropout_matrix_2)

attention_weight =torch.tensor(
        [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4833, 0.5167, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3190, 0.3408, 0.3402, 0.0000, 0.0000, 0.0000],
        [0.2445, 0.2545, 0.2542, 0.2468, 0.0000, 0.0000],
        [0.1994, 0.2060, 0.2058, 0.1935, 0.1953, 0.0000],
        [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]]
        )

dropout_mask=dropout(attention_weight)
print(dropout_mask)