import os
import sys
import easydict
from easydict import EasyDict

CONFIG=EasyDict()


#data param
CONFIG.DATA = EasyDict()
CONFIG.DATA.data_path = './dataset/RICO.pkl'
CONFIG.DATA.batch_size = 5
CONFIG.DATA.num_workers = 0

# model param
CONFIG.MODEL = EasyDict()
CONFIG.MODEL.model = 'Bert'
CONFIG.MODEL.epoch = 1

# optim param
CONFIG.OPTIM = EasyDict()
CONFIG.OPTIM.optim = 'AdamW'
CONFIG.OPTIM.lr = 1e-4
CONFIG.OPTIM.weight_decay = 0.5


CONFIG.EVAL = EasyDict()
CONFIG.EVAL.gen_t = 12



