from key_query_value import query_vector,key_vector,value_vector
import torch
query_vector = query_vector
key_vector = key_vector
value_vector = value_vector

print(query_vector)
print(key_vector)
print(value_vector)

# for calculating attention score dot product(query,keys)

query_vector_1=query_vector[1]
attention_score_1=torch.dot(query_vector_1,key_vector[1])
print(attention_score_1) # attention score for journey with repect to journey

attention_score_1_all=query_vector_1 @ key_vector.T 
print(attention_score_1_all)  # attention score for journey with respect to another queries

attention_scores = query_vector @ key_vector . T
print(attention_scores)   #attention scores for all the queries with respect to another queries
