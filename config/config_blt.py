import os
import sys
import easydict
from easydict import EasyDict

CONFIG=EasyDict()


#data param
CONFIG.DATA = EasyDict()
CONFIG.DATA.train_data_path = './data/RICO_train.pkl'
CONFIG.DATA.test_data_path = './data/RICO_test.pkl'
CONFIG.DATA.val_data_path = './data/RICO_val.pkl'
CONFIG.DATA.batch_size = 2
CONFIG.DATA.num_workers = 0

# model param
CONFIG.MODEL = EasyDict()
CONFIG.MODEL.model = 'BertForMaskedLM'
CONFIG.MODEL.epoch = 2
CONFIG.MODEL.loss = 'CrossEntropy'
CONFIG.MODEL.checkpoint_path = './checkpoint/blt_checkpoint.pt'

# optim param
CONFIG.OPTIM = EasyDict()
CONFIG.OPTIM.optim = 'AdamW'
CONFIG.OPTIM.lr = 1e-4
CONFIG.OPTIM.weight_decay = 0.5


CONFIG.EVAL = EasyDict()
CONFIG.EVAL.gen_t = 12
CONFIG.EVAL.eval_interval = 1


CONFIG.LOG = EasyDict()
CONFIG.LOG.output_path = './out/output.pkl'
CONFIG.LOG.log_path = './log/log.out'



