import torch

inputs = torch.tensor(
 [[0.43, 0.15, 0.89], # Your (x^1)
 [0.55, 0.87, 0.66], # journey (x^2)
 [0.57, 0.85, 0.64], # starts (x^3)
 [0.22, 0.58, 0.33], # with (x^4)
 [0.77, 0.25, 0.10], # one (x^5)
 [0.05, 0.80, 0.55]] # step (x^6)
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# print(attn_scores_2)

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
# print("Attention weights:", attn_weights_2_tmp)
# print("Sum:", attn_weights_2_tmp.sum())

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

# print("Attention weights:", attn_weights_2_naive)
# print("Sum:", attn_weights_2_naive.sum())

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
# print("Attention weights:", attn_weights_2)
# print("Sum:", attn_weights_2.sum())

query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
# print(context_vec_2)

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

# print(attn_scores)

attn_scores = inputs @ inputs.T
# print(attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
# print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# print("Row 2 sum:", row_2_sum)
# print("All row sums:", attn_weights.sum(dim=-1))

all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

# print("Previous 2nd context vector:", context_vec_2)

# 3.4.1

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
# print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
# print("keys.shape:", keys.shape)
# print("values.shape:", values.shape)

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
# print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# print(attn_weights_2)


context_vec_2 = attn_weights_2 @ values
# print(context_vec_2)

# 3.4.3 using classes
from selfAttentionClass import SelfAttention_v1

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# print(sa_v1(inputs))

from selfAttentionClass import SelfAttention_v2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# print(sa_v2(inputs))

# 3.5.1 Applying a causal attention mask

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
# print(mask_simple)

masked_simple = attn_weights*mask_simple
# print(masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
# print(masked_simple_norm)

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
# print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
# print(dropout(example))

# print(dropout(attn_weights))

# 3.5.3 Implementing a compact causal attention class
from selfAttentionClass import CausalAttention

batch = torch.stack((inputs, inputs), dim=0)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)

# 3.6.1 Stacking multiple single-head attention layers 
from selfAttentionClass import MultiHeadAttentionWrapper

torch.manual_seed(123)
context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
 d_in, d_out, context_length, 0.0, num_heads=2
)
context_vecs = mha(batch)
# print(context_vecs)
# print("context_vecs.shape:", context_vecs.shape)

# 3.6.2 Implementing multi-head attention with weight splits

a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
 [0.8993, 0.0390, 0.9268, 0.7388],
 [0.7179, 0.7058, 0.9156, 0.4340]],
 [[0.0772, 0.3565, 0.1479, 0.5331],
 [0.4066, 0.2318, 0.4545, 0.9737],
 [0.4606, 0.5159, 0.4220, 0.5786]]]])

# print(a @ a.transpose(2, 3))

first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
# print("First head:\n", first_res)
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
# print("\nSecond head:\n", second_res)

from selfAttentionClass import MultiHeadAttention

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
