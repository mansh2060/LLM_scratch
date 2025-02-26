'''
while dividing a attention score during normalization a scaling dot product of root under dimension of key is done
why sqrt of dimension ?
1.For stability in learning
2.To make the variance of the dot product stable
'''
import torch
import numpy as np
tensor = torch.tensor([0.1,-0.2,0.3,-0.2,0.5])
softmax_result = torch.softmax(tensor,dim=-1)
print(softmax_result)
#  tensor([0.1925, 0.1426, 0.2351, 0.1426, 0.2872]) this is perfectly normalized attention score

scaled_tensor = tensor * 8
softmax_result = torch.softmax(scaled_tensor,dim=-1)
print(softmax_result)
# tensor([0.0326, 0.0030, 0.1615, 0.0030, 0.8000]) this is the attention score after scalar multiplication which is completely peaky for some value and non peaky for other value

def compute_variance(dim,num_trials=1000):
    dot_products=[]
    scaled_dot_products=[]
    for _ in range(num_trials):
        q=np.random.randn(dim)
        k=np.random.randn(dim)
        dot_product=np.dot(q,k)
        dot_products.append(dot_product)
        scaled_dot_product=dot_product/np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)
    
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling  = np.var(scaled_dot_products)
    return variance_before_scaling,variance_after_scaling

variance_before_5,variance_after_5=compute_variance(5)
print(f"Variance_Before_5 {variance_before_5}")
print(f"Variance_After_5  {variance_after_5}")

variance_before_20,variance_after_20=compute_variance(20)
print(f"Variance_Before_5  {variance_before_20}")
print(f"Variance_After_5   {variance_after_20}")



