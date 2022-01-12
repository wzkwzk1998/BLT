import os
import sys
import argparse
import path
import tqdm
import random
import torch
import torch.nn as nn
from transformers.models.bert import modeling_bert

sys.path.append(path.Path(__file__).abspath().parent.parent)
sys.path.append(path.Path(__file__).abspath().parent)
from config.config_blt import CONFIG
from feeder import ricoDataset


from torch.utils.data import DataLoader
from model.BLTModel import BLTModel
from transformers import T5Model, T5Tokenizer, T5Config, AdamW
from transformers import BertModel, BertTokenizer, BertConfig



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
    if CONFIG.MODEL.model == 'BLT_MODEL':
        model = BLTModel()
    elif CONFIG.MODEL.model == 'T5':
        model = T5Model.from_pretrained('t5-base')
    elif CONFIG.MODEL.model == 'BERT':
        model = BertModel('bert-base-uncased')
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
        model_tokenizer = T5Tokenizer('t5-base')
    elif CONFIG.MODEL.model == 'Bert':
        model_config =  BertConfig()
        model_tokenizer = BertTokenizer('bert-base-uncased')

def mask_token(tokens, prob, idx):
    '''
    mask a token
    '''
    # TODO: implement mask token
    if prob < 0.15:
        if CONFIG.MODEL.model == 'bert':
            tokens[idx] = '[MASK]'
        elif CONFIG.MODEL.model == 'T5':
            tokens[idx] = '<extra_id_0>'


def random_mask(data):
    for sentence in data:
        type = random.randint(1,3)
        tokens= sentence.split()
        ele_num = len(tokens)               # element num per layout

        for i in range(ele_num):

            # mask obj class
            if type == 1:
                prob = random.random()
                mask_token(tokens, prob, i * 5)

            # mask coordinate param
            elif type == 2:
                prob_1 = random.random()
                mask_token(tokens, prob_1, i * 5 + 1)
                prob_2 = random.random()
                mask_token(tokens, prob_2, i * 5 + 2)
            
            # mask size param

            elif type == 3:
                prob_1 = random.random()
                mask_token(tokens, prob_1, i * 5 + 3)
                prob_2 = random.random()
                mask_token(tokens, prob_2, i * 5 + 4)



def train(model, optim, dataloader):
    for epoch in range(CONFIG.MODEL):
        #TODO: 写trian代码
        pass
        

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
        train(model, optim, dataloader)


    elif args.type == 'eval':
        # TODO: 写测试方法
        pass


    
        


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['train', 'eval'], help="mode, type is train or eval", default='train')
    parser.add_argument("--devices", type=str, help="gpu to use", default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    





