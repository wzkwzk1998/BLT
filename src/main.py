import os
import sys
import path
import argparse
import tqdm
import torch
import torch.nn as nn

sys.path.append(path.Path(__file__).abspath().parent.parent)
sys.path.append(path.Path(__file__).abspath().parent)
from config.config_blt import CONFIG
from feeder import ricoDataset

from torch.utils.data import DataLoader
from model.BLTModel import BLTModel
from transformers import T5Model, T5Tokenizer, T5Config, AdamW



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


def load_model(args):
    if CONFIG.MODEL.model == 'BLT_MODEL':
        model = BLTModel()
    elif CONFIG.MODEL.model == 'T5':
        model = T5Model.from_pretrained('t5-small')
    return model


def load_optim(model):
    print('[using optimizer] : {}'.format(CONFIG.OPTIM.optim))
    if CONFIG.OPTIM.optim == 'AdamW':
        optim = AdamW(model.parameters(), lr=CONFIG.OPTIM.lr)
    else:
        raise Exception('optimizer [{}] not implement yet'.format(CONFIG.OPTIM.optim))

    return optim

def load_data():
    data_loader =  DataLoader(ricoDataset.RicoDataset(CONFIG.DATA.data_path),
                            batch_size=CONFIG.DATA.batch_size,
                            shuffle=False,
                            num_workers=CONFIG.DATA.num_workers)

    return data_loader

def main():
    args = load_args()

    # load model
    model = load_model(args)
    dev = set_gpu(args.devices, model)

    #load optim
    optim = load_optim(model)
    data_loader = load_data()


    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    for data in data_loader:
        


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, help="epoch to run", default=50)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--type", type=str, choices=['train', 'eval'], help="mode, type is train or eval", default='train')
    parser.add_argument("--devices", type=str, help="gpu to use", default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    




