from ast import arg
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
from transformers import BertModel, BertTokenizer, BertConfig




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
    model = BLTModel(CONFIG.MODEL.model)
    return model


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
    elif CONFIG.MODEL.model == 'Bert':
        model_config =  BertConfig()
        model_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model_config, model_tokenizer

def mask_token(tokens, prob, idx, mask_idx_batch):
    '''
    mask a token
    '''
    # TODO: implement mask token
    if prob < 0.15:
        if CONFIG.MODEL.model == 'Bert':
            tokens[idx] = '[MASK]'
        elif CONFIG.MODEL.model == 'T5':
            tokens[idx] = '<extra_id_0>'
        mask_idx_batch.append(idx)


def random_mask(data):
    mask_idx = []
    for idx, sentence in enumerate(data):
        mask_idx_batch = []
        type = random.randint(1,3)
        tokens= sentence.split()
        assert(len(tokens) % 5 == 0)
        ele_num = int(len(tokens) / 5)               # element num per layout
        

        for i in range(ele_num):
            # mask obj class
            if type == 1:
                prob = random.random()
                mask_token(tokens, prob, i * 5, mask_idx_batch)

            # mask coordinate param
            elif type == 2:
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
                if CONFIG.MODEL.model == 'Bert':
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

def main():
    args = load_args()

    # load model
    model_config, model_tokenizer = get_config_and_tokenizer()
    model = load_model(args, model_config)
    dev = set_gpu(args.devices, model)

    #load optim 
    optim = load_optim(model)
    data_loader = load_data()

    if args.type == 'train':
        
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
                
                print(output.shape)
                
                # TODO: implement loss calculation

                #backward
                optim.zero_grad()
                # loss.backward()
                optim.step()
                break

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
    





