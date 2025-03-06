import re
from Tokenizer import SimpleTokenizerV2
from importlib.metadata import version
import tiktoken
from torch.utils.data import DataLoader
from CustomDataloader import GPTDatasetV1
import torch


# Simple encoding

file_path = "Chapter2\\the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as f:
 raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

for i, item in enumerate(list(vocab.items())[-5:]):
    #print(item)
    pass


# Byte pair encoding

tokenizer = tiktoken.get_encoding("gpt2")

text = (
 "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
 "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

strings = tokenizer.decode(integers)
# print(strings)

# Example of unknown word

word = "Akwirw ier"
integers = tokenizer.encode(word, allowed_special={"<|endoftext|>"})
# print(integers)
strings = tokenizer.decode(integers)
#print(strings)

# Data sampling with a sliding window

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,
    num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
    )
    return dataloader

file_path = "Chapter2\\the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=4, stride=4,
 shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# 2.8 Encoding word positions

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


max_length = 4
dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=max_length,
 stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

token_embeddings = token_embedding_layer(inputs)


context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)