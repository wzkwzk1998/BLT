from ast import arg
from functools import total_ordering
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
from math import gamma
import numpy as np
import os
from statistics import mode
import sys
import argparse
from numpy import iterable
import path
import random
import math
import torch
import torch.nn as nn
import datetime
import json


from tensorboardX import SummaryWriter
from transformers.models.bert import modeling_bert
from transformers import T5ForConditionalGeneration, T5Model, T5Tokenizer, T5Config, AdamW
from transformers import BertModel, BertTokenizer, BertConfig, BertForMaskedLM
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.append(path.Path(__file__).abspath().parent.parent)
sys.path.append(path.Path(__file__).abspath().parent)
from utils.metric_tools import cal_alignment, cal_IoU, cal_overlap, overlap
from config.config_blt import CONFIG
from feeder import ricoDataset
from model.BLTModel import BLTModel



def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['train', 'eval'], help="mode, train or eval", default='train')
    parser.add_argument("--devices", type=str, help="gpu to use", default="")
    parser.add_argument("--debug", default=False, action='store_true')
    parser.add_argument("--weight_path", default="")
    args = parser.parse_args()
    return args

def set_gpu(devices:str, model):

    os.environ["CUDA_VISIBLE_DEVICES"]=devices

    #cpu
    if devices == "":
        print('[using cpu]')
        model.to('cpu')
        dev = 'cpu'
        return model, dev

    #gpu
    devices_list = devices.split(",")
    if len(devices_list) != 0:
        gpus = [int(i) for i in range(len(devices_list))]
        print('[using gpu]')
        dev = 'cuda:0'
        model.to('cuda')

    
    if len(gpus) > 0 and torch.cuda.device_count() > 1:
        print('[user gpu]: {}'.format(devices))
        model=nn.DataParallel(model, device_ids=gpus)

    return model, dev


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


def load_data(data_path, args):
    data_loader =  DataLoader(ricoDataset.RicoDataset(data_path, args.debug),
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
    if prob < 0.2:
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
    result = []
    for idx,sentence in enumerate(data):
        tokens = sentence.split()
        assert(len(tokens) % 5 == 0)
        ele_num = int(len(tokens) / 5)
        for i in range(ele_num):
            for j in range(1,5):
                if CONFIG.MODEL.model == 'BertForMaskedLM':
                    tokens[i * 5 + j] = '[MASK]'
                elif CONFIG.MODEL.model == 'Bert':
                    tokens[i * 5 + j] = '[MASK]'
                elif CONFIG.MODEL.model == 'T5':
                    tokens[i * 5 + j] = '<extra_id_0>'
                else:
                    raise Exception("unknow model")
            
        sentence_masked = ' '.join(tokens)
        result.append(sentence_masked)
    
    return result
        

def num_to_word(pred):

    seq = []
    
    for num in pred.squeeze(0):
        assert 0 <= num 
        assert num <= 127
        seq.append(str(num.item()))

    return ' '.join(seq)


def mask_by_mask_ids(preds, mask_ids):
    '''
    在这个函数中，我们需要我们mask_ids来指示哪些需要mask掉
    '''
    if CONFIG.MODEL.model == 'Bert' or CONFIG.MODEL.model == 'BertForMaskedLM':
        mask_token_idx = 103
    elif CONFIG.MODEL.model == 'T5':
        mask_token_idx = 32099

    for seq, mask in zip(preds, mask_ids):
        seq[mask_ids] = mask_token_idx

    return preds   


def recover_class_by_label(preds, labels):
    for pred, label in zip(preds, labels):
        for idx in range(int((len(pred) - 2) / 5)):             # minus to for <bos> and <eos>
            pred[idx * 5 + 1] = label[idx * 5 + 1]              # ADD ONE FOR <BOS>
    
    return preds

def truncate_by_label(preds, labels):
    result = []
    for pred, label in zip(preds, labels):
        # print('pred shape : {}'.format(pred.shape))
        tokens_label = label.split()
        result.append(pred[1:len(tokens_label)+1].tolist())
    return result

def seq_to_num_and_store(seqs, labels):
    result = []
    origin = []
    bad_layout = 0
    for seq, labels in zip(seqs, labels):
        result_layout = []
        origin_layout = []
        tokens = seq.split()
        origin_tokens = labels.split()
        if len(tokens) % 5 != 0:                     # 如果出现了像'.'这样的东西导致长度缩短，那么整个layout直接不要
            bad_layout += 1
            continue
        box_num = int(len(tokens) / 5)
        for i in range(box_num):
            result_box = []
            origin_box = []
            for j in range(5):
                if not tokens[i * 5 + j].isdigit() or int(tokens[i * 5 + j]) < 0:
                    break
                result_box.append(int(tokens[i * 5 + j]))
                origin_box.append(int(origin_tokens[i * 5 + j]))
            if len(result_box) == 5:
                result_layout.append(result_box)
                origin_layout.append(origin_box)
        if len(result_layout) > 0:
            result.append(result_layout)
            origin.append(origin_layout)
        
    return result, origin
    
        
def train(args, model, model_tokenizer, dev, optim, train_data_loader, test_data_loader, writer=None):

    for epoch in range(CONFIG.MODEL.num_epoch):                
        with tqdm(iterable=train_data_loader) as t:
            start_time = datetime.datetime.now()
            loss_list = []
            model.train()
            for data in train_data_loader:
                t.set_description_str(f"\33[36m [Epoch {epoch + 1:04d}]")
                label = data[:]
                data, mask_ids = random_mask(data)
                
                # print(data)
                # print(label)
                # print(mask_ids)

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
                loss = output.loss.mean()
                loss.backward()
                optim.step()

                loss_list.append(loss.item())
                cur_time = datetime.datetime.now()
                delta_time = cur_time - start_time

                t.set_postfix_str(f"train_loss:{sum(loss_list) / len(loss_list):.6f}, time:{delta_time}\33[0m")
                t.update()

            if not os.path.exists(CONFIG.LOG.log_dir):
                os.makedirs(CONFIG.LOG.log_dir)
            with open(os.path.join(CONFIG.LOG.log_dir, TIME_PREFIX + '.log'), 'a') as f:
                f.write('epoch {}, train loss : {}\n'.format(epoch, sum(loss_list) / len(loss_list)))
                
            if epoch % CONFIG.EVAL.eval_interval == 0:
                test(args, model, model_tokenizer=model_tokenizer, dev=dev, test_data_loader=test_data_loader, writer=writer, epoch=epoch)
            if epoch % CONFIG.EVAL.save_interval == 0:
                if not os.path.exists(CONFIG.MODEL.checkpoint_dir):
                    os.makedirs(CONFIG.MODEL.checkpoint_dir)
                torch.save(model.state_dict(), os.path.join(CONFIG.MODEL.checkpoint_dir, TIME_PREFIX+'.pt'))

            if args.type == 'train' and not args.debug:
                writer.add_scalar('train loss', sum(loss_list)/len(loss_list), epoch)
            
            t.update()
        

def test(args, model, model_tokenizer, dev, test_data_loader, writer=None, epoch=0):

    model.eval()
    loss_list = []
    result = []
    origin = []
    iou_list = []
    overlap_list = []
    alignment_list = []
    
    with tqdm(iterable=test_data_loader) as t:
        for data in test_data_loader:
            t.set_description_str(f"[Eval]")
            label = data[:]
            data = mask_cor_and_size(data)

            inputs = model_tokenizer(data, return_tensors='pt', padding=True)
            label_out = model_tokenizer(label, return_tensors='pt', padding=True)

            input_ids = inputs['input_ids'].to(dev)
            attention_mask = inputs['attention_mask'].to(dev)
            token_type_ids = inputs['token_type_ids'].to(dev)
            labels = label_out['input_ids'].to(dev)

            # ele_num = int(((input_ids.shape[1]) - 2) / 5)                      # minus two for <bos> and <eos>

            # groups = [[0, 1, 2], [0, 3, 4]]                                      # don't mask class and size parameter


            # TODO: 先完成非iterative 再加上iterative                    
            # generate cor and size
            # for group in groups:
            #     type_not_mask = [int(i * 5 + j + 1) for i in range(int(ele_num)) for j in group]
            #     type_not_mask.append(0)                                             # don't mask <bos>
            #     type_not_mask.append((ele_num * 5 + 1))                             # don't mask <eos>
            #     #generate size
            #     for t in range(1, int(CONFIG.EVAL.gen_t / 2)  + 1):             # t from range  1 ->  T/2       
            #         output = model(input_ids, attention_mask, token_type_ids)
            #         gamma = (CONFIG.EVAL.gen_t - 2 * t) / CONFIG.EVAL.gen_t
            #         mask_num = math.ceil(gamma * ele_num * 2)
                    
            #         logit, pred = torch.max(output.logits, dim=2)        
                    
            #         logit[:, type_not_mask] = float(sys.maxsize) 
            #         mask_ids = torch.argsort(logit, dim=1)[:, :mask_num]           # which token to re-masked, coordinate group
                
            #         pred_recovered = recover_class_by_label(pred, labels)
            #         pred_mask = mask_by_mask_ids(pred_recovered, mask_ids)

            #         # print("mask_num : {}".format(mask_num))
            #         # print("input_ids shape : {}".format(input_ids.shape))
            #         # print("input_ids is : {}".format(input_ids))
            #         # print("pred_mask shape :{}".format(pred_mask.shape))
            #         # print("pred_mask is : {}".format(pred_mask))

            #         input_ids = pred_mask
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            
            loss = output.loss.mean()
            loss_list.append(loss.item())

            logit, pred = torch.max(output.logits, dim=2)
            pred = truncate_by_label(pred, label)
            seq = model_tokenizer.batch_decode(pred)
            result_batch, origin_batch = seq_to_num_and_store(seq, label)
            result = result + result_batch
            origin = origin + origin_batch

            # print(pred)
            # print(result)

            batch_iou = cal_IoU(result_batch)
            batch_overlap = cal_overlap(result_batch)
            batch_alignment = cal_alignment(result_batch)

            # print("batch_iot : {}".format(batch_iou))
            # print("batch_overlap : {}".format(batch_overlap))
            # print("batch_alignment : {}".format(batch_alignment))

            iou_list.append(batch_iou)
            overlap_list.append(batch_overlap)
            alignment_list.append(batch_alignment)

            t.set_postfix_str(f"eval_loss:{sum(loss_list) / len(loss_list):.6f}")
            t.update()
        
    if not os.path.exists(CONFIG.LOG.log_dir):
        os.makedirs(CONFIG.LOG.log_dir)   
    with open(os.path.join(CONFIG.LOG.log_dir, TIME_PREFIX + '.log'), 'a') as f:
        f.write('iou : {}\n'.format(sum(iou_list) / len(iou_list)))
        f.write('alignment : {}\n'.format(sum(alignment_list) / len(alignment_list)))
        f.write('overlap : {}\n'.format(sum(overlap_list) / len(overlap_list)))
        f.write('eval loss : {}\n'.format(sum(loss_list) / len(loss_list)))
        print('eval loss : {}'.format(sum(loss_list) / len(loss_list)))
        print('alignment : {}'.format(sum(alignment_list) / len(alignment_list)))
        print('overlap : {}'.format(sum(overlap_list) / len(overlap_list)))
        print('iou : {}'.format(sum(iou_list) / len(iou_list)))

    if args.type == 'train' and not args.debug:
        writer.add_scalar('eval loss', sum(loss_list) / len(loss_list), epoch)
        writer.add_scalar('overlap', sum(overlap_list) / len(overlap_list), epoch)
        writer.add_scalar('alignment', sum(alignment_list)/len(alignment_list), epoch)
        writer.add_scalar('iou', sum(iou_list) / len(iou_list), epoch)

    json_dict = {}
    json_dict['origin'] = origin
    json_dict['generation'] = result
    

    if not os.path.exists(CONFIG.LOG.output_dir):
        os.makedirs(CONFIG.LOG.output_dir)

    with open(os.path.join(CONFIG.LOG.output_dir, TIME_PREFIX + '.json'), 'w') as fp:
        json.dump(json_dict, fp)

    t.update()

    
def main():
    args = load_args()

    # tensorboardX
    if args.type == 'train' and not args.debug:
        writer = SummaryWriter(os.path.join(CONFIG.LOG.tensorboard_dir, TIME_PREFIX + '.log'))
    else:
        writer = None
    # load model
    model_config, model_tokenizer = get_config_and_tokenizer()
    model = load_model(args, model_config)
    # load dev
    model, dev = set_gpu(args.devices, model)
    # load weight
    if args.weight_path != '':
        print('load weight from {}'.format(args.weight_path))
        model.load_state_dict(torch.load(args.weight_path))

    # load optim 
    optim = load_optim(model)

    
    # trian or eval
    if args.type == 'train' and CONFIG.MODEL.model == 'BertForMaskedLM':

        train_data_loader = load_data(CONFIG.DATA.train_data_path, args)
        val_data_loader = load_data(CONFIG.DATA.val_data_path, args)
        train(args, model, model_tokenizer=model_tokenizer, 
            dev=dev, optim=optim, train_data_loader=train_data_loader, test_data_loader=val_data_loader, writer=writer)
                
    elif args.type == 'eval' and CONFIG.MODEL.model == 'BertForMaskedLM':
        
        test_data_loader = load_data(CONFIG.DATA.test_data_path, args)
        test(args, model, model_tokenizer, dev, test_data_loader=test_data_loader, writer=writer)

            
if __name__ == '__main__':
    TIME_PREFIX = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    main()
    





