
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


def main():
    assert(False)


seq = ['I', 'am','not','human']
print(seq[1:-1])
main()

