
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model



seq = 'I am a big big big big big pig'
inputs = T5Tokenizer.get_special_tokens_mask([1,1,1,1,1])