from ast import arg
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
from math import gamma
import os
from statistics import mode
import sys
import argparse
import path
import tqdm
import random
import math
import torch
import torch.nn as nn
from transformers.models.bert import modeling_bert

sys.path.append(path.Path(__file__).abspath().parent.parent)
sys.path.append(path.Path(__file__).abspath().parent)
from config.config_blt import CONFIG
from feeder import ricoDataset


from torch.utils.data import DataLoader
from model.BLTModel import BLTModel

from transformers import T5ForConditionalGeneration, T5Model, T5Tokenizer, T5Config, AdamW
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM




def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['train', 'eval'], help="mode, train or eval", default='train')
    parser.add_argument("--devices", type=str, help="gpu to use", default="")
    args = parser.parse_args()
    return args

def set_gpu(devices:str, model):

    os.environ["CUDA_VISIBLE_DEVICES"]=devices

    #cpu
    if devices == "":
        print('[using cpu]')
        model.to('cpu')
        dev = 'cpu'
        return

    #gpu
    devices_list = devices.split(",")
    if len(devices_list) != 0:
        gpus = [int(gpu_str) for gpu_str in devices_list]
        print('[using gpu]')
        dev = 'cuda:0'
        model.to('cuda')

    if len(gpus) > 0 and torch.cuda.device_count() > 1:
        model=nn.DataParallel(model, device_ids=gpus)

    return dev


def load_model(args, model_config):
    if CONFIG.MODEL.model == 'BertForMaskedLM':
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    else:
        model = BLTModel(CONFIG.MODEL.model)
    return model

def load_loss_func():
    if CONFIG.MODEL.loss == 'CrossEntropy':
        loss_func = nn.CrossEntropyLoss()
    
    return loss_func
def load_optim(model):
    print('[using optimizer] : {}'.format(CONFIG.OPTIM.optim))
    if CONFIG.OPTIM.optim == 'AdamW':
        optim = AdamW(model.parameters(), lr=CONFIG.OPTIM.lr)
    elif CONFIG.OPTIM.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=CONFIG.OPTIM.lr)
    else:
        raise Exception('optimizer [{}] not implement yet'.format(CONFIG.OPTIM.optim))

    return optim


def load_data():
    data_loader =  DataLoader(ricoDataset.RicoDataset(CONFIG.DATA.data_path),
                            batch_size=CONFIG.DATA.batch_size,
                            shuffle=False,
                            num_workers=CONFIG.DATA.num_workers)

    return data_loader


def get_config_and_tokenizer():
    if CONFIG.MODEL.model == 'T5':
        model_config = T5Config()
        model_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    elif CONFIG.MODEL.model == 'Bert' or CONFIG.MODEL.model == 'BertForMaskedLM':
        model_config =  BertConfig()
        model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model_config, model_tokenizer


def mask_token(tokens, prob, idx, mask_idx_batch):
    '''
    mask a token
    '''
    if prob < 0.15:
        if CONFIG.MODEL.model == 'BertForMaskedLM':
            tokens[idx] = '[MASK]'
        if CONFIG.MODEL.model == 'Bert':
            tokens[idx] = '[MASK]'
        elif CONFIG.MODEL.model == 'T5':
            tokens[idx] = '<extra_id_0>'
        mask_idx_batch.append(idx)


def random_mask(data):
    mask_idx = []
    for idx, sentence in enumerate(data):
        mask_idx_batch = []
        type = random.randint(2,3)
        tokens= sentence.split()
        assert(len(tokens) % 5 == 0)
        ele_num = int(len(tokens) / 5)               # element num per layout
        

        for i in range(ele_num):

            # mask coordinate param
            if type == 2:
                prob_1 = random.random()
                mask_token(tokens, prob_1, i * 5 + 1, mask_idx_batch)
                prob_2 = random.random()
                mask_token(tokens, prob_2, i * 5 + 2, mask_idx_batch)
            
            # mask size param
            elif type == 3:
                prob_1 = random.random()
                mask_token(tokens, prob_1, i * 5 + 3, mask_idx_batch)
                prob_2 = random.random()
                mask_token(tokens, prob_2, i * 5 + 4, mask_idx_batch)

        sentence_masked = ""
        for token in tokens:
            sentence_masked = sentence_masked + ' ' + token
        data[idx] = sentence_masked.lstrip()

        mask_idx.append(mask_idx_batch)

    return data, mask_idx

def mask_cor_and_size(data):
    '''
    mask all coordinate and size parameter in a batch
    '''
    for idx,sentence in enumerate(data):
        tokens = sentence.split()
        assert(len(tokens) % 5 == 0)
        ele_num = int(len(tokens) / 5)
        for i in range(ele_num):
            for j in range(1,5):
                if CONFIG.MODEL.model == 'BertForMaskedLM':
                    tokens[i * 5 + j] == '[MASK]'
                elif CONFIG.MODEL.model == 'Bert':
                    tokens[i * 5 + j] = '[MASK]'
                elif CONFIG.MODEL.model == 'T5':
                    tokens[i * 5 + j] = '<extra_id_0>'
                else:
                    raise Exception("unknow model")
            
        sentence_masked = ' '.join(tokens)
        data[idx] = sentence_masked
    
    return data
        

def num_to_word(pred):

    seq = []
    
    for num in pred.squeeze(0):
        assert 0 <= num 
        assert num <= 127
        seq.append(str(num.item()))

    return ' '.join(seq)


def mask_by_pred(sequence, mask_ids):
    '''
    在这个函数中，我们需要我们mask_ids来指示哪些需要mask掉
    '''
    if CONFIG.MODEL.model == 'Bert' or CONFIG.MODEL.model == 'BertForMaskedLM':
        mask_token = '[MASK]'
    elif CONFIG.MODEL.model == 'T5':
        mask_token = '<extra_id_0>'

    print("mask_ids shape: {}".format(mask_ids.shape))
    result = []
    for seq, mask in zip(sequence, mask_ids):
        tokens = seq.split()
        tokens[mask] = mask_token
        result.append(' '.join(tokens))

    return result    


def recover_seq(sequence, labels):
    # TODO: 完成将对应的class重新赋值给seq的方法
    pass

  

def main():
    args = load_args()

    # load model
    model_config, model_tokenizer = get_config_and_tokenizer()
    model = load_model(args, model_config)
    dev = set_gpu(args.devices, model)

    # load optim 
    optim = load_optim(model)
    # load dataloader
    data_loader = load_data()
    # load loss function
    loss_func = load_loss_func()


    if args.type == 'train' and CONFIG.MODEL.model == 'BertForMaskedLM':
        
        for epoch in range(CONFIG.MODEL.epoch):
            model.train()
            for data in data_loader:
                label = data[:]
                data, mask_ids = random_mask(data)

                print(data)
                print(label)
                print(mask_ids)

                inputs = model_tokenizer(data, return_tensors='pt', padding=True)
                label_out = model_tokenizer(label, return_tensors='pt', padding=True)
                # to dev
                input_ids = inputs['input_ids'].to(dev)
                attention_mask = inputs['attention_mask'].to(dev)
                token_type_ids = inputs['token_type_ids'].to(dev)
                labels = label_out['input_ids'].to(dev)

                
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

                optim.zero_grad()
                # TODO：there may be a problem in this loss, need to double check it. 
                output.loss.backward()
                optim.step()
                print(output.loss)

            


    if args.type == 'train' and CONFIG.MODEL.model != 'BertForMaskedLM':
        
        for epoch in range(CONFIG.MODEL.epoch):
            model.train()
            for data in data_loader:
                label = data[:]            # label remain the same as raw data
                data,mask_ids  = random_mask(data)

                print(data)
                print(label)
                print(mask_ids)

                inputs = model_tokenizer(data, return_tensors='pt', padding=True)
                label_out = model_tokenizer(label, return_tensors='pt', padding=True)

                # to dev
                inputs_ids = inputs['input_ids'].to(dev)
                label_out_ids = label_out['input_ids'].to(dev)

                #forward
                output = model(input_ids=inputs_ids)                
                # TODO: implement loss calculation
                loss = loss_func(output.transpose(1,2), label_out_ids)
                #backward
                optim.zero_grad()
                # loss.backward()
                optim.step()
                break
    
    elif args.type == 'eval' and CONFIG.MODEL.model == 'BertForMaskedLM':
        # TODO: 测试方法的生成是生成一个batch还是一个一个生成有待于double check
        print('BertForMaskedLM')
        model.eval()
        loss_list = []
        result = []
        
        for data in data_loader:
            label = data[:]
            data = mask_cor_and_size(data)

            inputs = model_tokenizer(data, return_tensors='pt', padding=True)
            label_out = model_tokenizer(label, return_tensors='pt', padding=True)

            # to dev
            input_ids = inputs['input_ids'].to(dev)
            attention_mask = inputs['attention_mask'].to(dev)
            token_type_ids = inputs['token_type_ids'].to(dev)
            labels = label_out['input_ids'].to(dev)

            ele_num = int((len(input_ids[1]) - 2) / 5)                      # minus two for <bos> and <eos>
            cardinal_for_size = cardinal_for_cor  = ele_num * 2
            
            for t in range(1, int(CONFIG.EVAL.gen_t / 2)  + 1):             # t from range  1 ->  T/2       
                output = model(input_ids, attention_mask, token_type_ids)
                gamma = (CONFIG.EVAL.gen_t - 3 * t) / CONFIG.EVAL.gen_t
                mask_num = math.ceil(gamma * cardinal_for_cor)
                
                logit, pred = torch.max(output.logits[:, 1:-1, :], dim=2)        # ignore <bos> and <eos>

                if mask_num <= 0.0 :
                    break 
                else:
                    group_position = [0, 3, 4]                                      # don't mask class and size parameter
                    type_mask = [int(i * 5 + j) for i in range(int(ele_num)) for j in group_position]
                    logit[:, type_mask] = float(sys.maxsize) 
                    mask_ids = torch.argsort(logit, dim=1)[:, :mask_num]           # which token to re-masked, coordinate group
                    
                    # id to seq
                    seq = model_tokenizer.batch_decode(pred)
                    print(seq)
                    seq_masked = mask_by_pred(seq, mask_ids)
                    seq_masked = recover_seq(seq_masked, label)
                    
                    

                    inputs_uni = model_tokenizer(seq_masked, return_tensors='pt', padding=True)

                    # to dev
                    input_ids = inputs_uni['input_ids'].to(dev)
                    attention_mask = inputs_uni['attention_mask'].to(dev)
                    token_type_ids = inputs_uni['token_type_ids'].to(dev)


            print(pred.shape)        


    elif args.type == 'eval':
        # TODO: 写测试方法, 记得把数据集划分为测试集和训练集
        print('eval')
        model.eval()
        for data in data_loader:
            # process data
            label = data[:]
            data = mask_cor_and_size(data)
            
            print(data)

            inputs = model_tokenizer(data, return_tensors='pt', padding=True)
            label_out = model_tokenizer(label, return_tensors='pt', padding=True)

            # to dev
            inputs_ids = inputs['input_ids'].to(dev)
            label_out_ids = label_out['input_ids'].to(dev)


            for input_ids in inputs_ids:                        # 将batch_size 中的每一个batch分开处理，要不长度不统一会出问题
                input_ids = input_ids.unsqueeze(0)
                print(input_ids.shape)
                ele_num = len(input_ids) / 5

                # generation for size
                for t in range(int(CONFIG.EVAL.gen_t / 2)):
                    output = model(input_ids)
                    print(output.shape)
                    gamma = (CONFIG.EVAL.gen_t - 3 * t) / CONFIG.EVAL.gen_t
                    mask_num = math.ceil(gamma * (ele_num * 2))
                    print(mask_num)
                    
                    mmax, pred = torch.max(output[:, 1:-1, :], dim=2)        # ignore <bos> and <eos>
                    mask_ids = torch.argsort(mmax, dim=1)[:mask_num]
                    nxt_setence = num_to_word(pred)
                    print(nxt_setence)

                    return 

            
if __name__ == '__main__':
    main()
    





