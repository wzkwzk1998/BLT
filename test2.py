
from numpy import argmax
import torch

from transformers import DebertaTokenizer, T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM



# seq = 'I am [MASK] big [MASK] big [MASK] big pig'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# inputs = tokenizer(seq)
# inputs_ids = inputs['input_ids']
# print(inputs_ids)

# seq = tokenizer.decode(inputs_ids)
# print(seq)


# model_1 = BertModel.from_pretrained('bert-base-uncased')
# model_2 = BertForMaskedLM.from_pretrained('bert-base-uncased')
# print(model_2)



# a = torch.rand([1,3,5])
# a = torch.tensor([[2,3,6,8,5,9]])

# print(a)

# b,c = torch.max(a, dim = 2)
# print(b)
# print(c)


# c = torch.argsort(a, dim=1)
# print(c)


# def main():
#     assert(False)


# seq = ['I', 'am','not','human']
# print(seq[1:-1])
# main()


# from transformers import BertTokenizer, BertForMaskedLM
# import torch

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# seq = 'oh my god'
# inputs = tokenizer(seq, return_tensors='pt')
# print(inputs)

# seq = ["The capital of France is [MASK].", "The capital of France is [MASK]."]
# seq_label = ["The capital of France is Paris.", "The capital of France is Paris."]

# inputs = tokenizer(seq, return_tensors="pt")
# labels = tokenizer(seq_label, return_tensors="pt")["input_ids"]

# print(inputs['input_ids'])
# seq_dec = tokenizer.batch_decode(inputs['input_ids'])
# print(seq_dec)
# print(labels)

# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# print(loss)

# logits = outputs.logits
# print(logits)


import torch
import sys



# a = torch.tensor([[[1.0,2.0,3.0,4.0,5.0]]]).transpose(1,2)

# b = torch.tensor([[1]])


# loss = torch.nn.CrossEntropyLoss(ignore_index=0)
# output = loss(a, b)
# print(output)

# print(sys.maxsize)
# print(float(sys.maxsize))
# a = torch.tensor([[1, 2, 3], [4, 5, 6]])


# b = torch.rand([2,3])


# for i, j in zip(a, b):
#     print(i)
#     print(j)
    


# seq = 'oh my god'

b = 'aaa'


for i in range(10):
    num = 1

print(num)



a = [1,2,3,4,5,6,7]
b = [11,22,33,44,55,66,77]
idx = [1,4,7,9]

print(a[idx])









