import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from transformers import T5ForConditionalGeneration, T5Model, T5Tokenizer, T5Config, AdamW
from transformers import BertModel, BertTokenizer, BertConfig




class BLTModel(nn.Module):
    def __init__(self, base_model) -> None:
        super().__init__()
        
        if base_model == 'Bert':
            self.base_model = BertModel.from_pretrained('bert-base-uncased')
        elif base_model == 'T5':
            self.base_model = T5Model.from_pretrained('t5-base')
        else:
            raise Exception("model {} has not been implemented yet".format(base_model))

        self.predict_head = nn.Linear(in_features=768, out_features=128, bias=True)


    def forward(self, input_ids):
        output_hidden = self.base_model(input_ids).last_hidden_state
        output = self.predict_head(output_hidden)
        return output
        


    