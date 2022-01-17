
from lib2to3.pgen2 import token
from unittest import skip
from numpy import argmax
from torch import tensor
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


from transformers import BertTokenizer, BertForMaskedLM, T5Tokenizer
import torch

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

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

# a = torch.tensor([[1021, 1014, 1014, 1014, 1012,  102, 1021, 1012,  103, 1064],
#         [1021, 1014, 1014, 1014, 1021, 1021, 1015, 1012,  103, 1012]])
# b = tokenizer.batch_decode(a)

# a = ['I am a big big pig', 'I am a big big big , pig pig pig pig pig']
# b = tokenizer(a)
# seq = tokenizer.batch_decode(b.input_ids)
# print(b)
# print(seq)






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

# b = '<extra_id_99>'

# inputs = tokenizer(b, padding=True)
# print(inputs)
# input_ids = [[101,103,1111,102]
# seq = tokenizer.decode(input_ids, skip_special_tokens=True)
# print(seq)
# seq = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False)
# print(inputs)
# print(seq)

# a = ["I am a big pig", "I am a big big pig pig pig pig pig"]
# inputs = tokenizer(a, return_tensors='pt', padding=True)
# output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, token_type_ids=inputs.token_type_ids)
# print(inputs)
# print(output.logits[0][7])

# for i in range(10):
#     num = 1

# print(num)



# a = [1,2,3,4,5,6,7]
# idx = [1,4,5]

# print(a[idx])


# def recover_class_by_label(preds, labels):
#     # TODO: 完成将对应的class重新赋值给seq的方法

#     for pred, label in zip(preds, labels):
#         for idx in range(int(len(pred) / 5)):
#             print(idx)
#             pred[idx * 5] = label[idx * 5]
    
#     return preds

# a = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
# b = torch.tensor([[11,12,13,14,15],[16,17,18,19,20]])
# a = recover_class_by_label(a, b)
# print(a)



# b = torch.max(a, dim=0)
# print(b)














