import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

RICO_LABELS_LOWER = [
        'text', 
        'image', 
        'icon', 
        'list item', 
        'text button', 
        'toolbar', 
        'web view', 
        'input',
        'card',
        'advertisement', 
        'background image', 
        'drawer', 
        'radio button', 
        'checkbox', 
        'multi-tab', 
        'pager indicator', 
        'modal', 
        'on/off switch', 
        'slider', 
        'map view', 
        'button bar', 
        'video', 
        'bottom navigation', 
        'number stepper', 
        'date picker'
]

class LayoutDataset(Dataset):
    def __init__(self, cfg, dataset, real=False):

        self.dataset = dataset
        self.max_ele_num = cfg.max_ele_num
        self.max_label_num = cfg.max_label_num
        self.real = real

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        fn = self.dataset[idx]["fn"]

        if self.real:
            label = self.dataset[idx]["label"]
            for i in range(len(label)):
                if label[i]>self.max_label_num:
                    label[i] = self.max_label_num
                if label[i]<1:
                    label[i] = 1
            box = self.dataset[idx]["box"] 
        else:
            label = self.dataset[idx]["pred_label"]
            for i in range(len(label)):
                if label[i]>self.max_label_num:
                    label[i] = self.max_label_num
                if label[i]<1:
                    label[i] = 1
            box = self.dataset[idx]["pred_box"] 
            
        pad_label, pad_box, label_mask = self.padding_data(label,box)

        sample = {'fn':fn, 'label':pad_label, 'box':pad_box, 'label_mask':label_mask}
        return sample
    
    def padding_data(self, label, box):
        pad_label = np.zeros([self.max_ele_num+2], dtype='int64')
        pad_box = np.zeros([self.max_ele_num+2, 4], dtype='float32')
        label_mask = np.zeros([self.max_ele_num+2], dtype='int64')

        for i in range(len(label)+2):
            label_mask[i] = 1
        
        # sos label is the max label index +1
        # eos label is the max label index +2
        pad_label[0] = self.max_label_num + 1
        pad_label[1:len(label)+1] = np.array(label)
        pad_label[len(label)+1] = self.max_label_num + 2

        pad_box[1:len(box)+1] = np.array(box)

        return pad_label, pad_box, label_mask


class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer('token_mask', token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
            ), num_layers=num_layers)

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class FidNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        d_model = cfg.d_model #256
        nhead = cfg.n_heads #4
        num_layers = cfg.n_layers_encoder #4
        max_bbox = cfg.max_ele_num + 2 #20
        num_label = cfg.max_label_num + 3 #25

        # encoder
        self.emb_label = nn.Embedding(num_label, d_model)
        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(d_model=d_model,
                                                    dim_feedforward=d_model // 2,
                                                    nhead=nhead, num_layers=num_layers)

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=d_model // 2)
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out_cls = nn.Linear(d_model, num_label)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask):
        b = self.fc_bbox(bbox)
        l = self.emb_label(label)
        x = self.enc_fc_in(torch.cat([b, l], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0]

    def forward(self, bbox, label, padding_mask):
        
        B, N, _ = bbox.size()
        x = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)

        # logit_cls: [M, L]    bbox_pred: [M, 4]
        logit_cls = self.fc_out_cls(x)
        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, logit_cls, bbox_pred



def parsing_text_to_label_set(text):

    text = text.replace('.', '')
    text = text.replace(',', '')

    label_set = []
    label_with_number = []

    word_list = text.split(' ')
    word_list_len = len(word_list)

    num_flag = False
    number = 0

    i = 0
    while i < word_list_len:

        if word_list[i].isdigit():
            num_flag = True
            number = int(word_list[i])
            i += 1
            continue
        if num_flag == True:
            # for label with two words
            if i < (word_list_len - 1):
                words = word_list[i].lower() + ' ' + word_list[i+1].lower()
                if words in RICO_LABELS_LOWER: 
                    index = RICO_LABELS_LOWER.index(words) + 1
                    if index not in label_set:
                        label_set.append(index)
                    for j in range(number):
                        label_with_number.append(index)
                    num_flag = False
                    i += 2
                    continue 
        
            if word_list[i].lower() in RICO_LABELS_LOWER:
                index = RICO_LABELS_LOWER.index(word_list[i].lower()) + 1
                if index not in label_set:
                    label_set.append(index)
                for j in range(number):
                    label_with_number.append(index)
                num_flag = False
                i += 1 
                continue
        i += 1 

    return label_set, label_with_number

def _test_parsing_text_to_label_set():

    with open('../data/layout_nl10.txt','r', encoding='utf-8') as f:
        nl_list = f.read().rstrip('\n').split('\n')

    for item in nl_list:
        item = item.split('\t')
        fn = item[0]
        number = item[2]
        text = item[1]
        _, label_with_number = parsing_text_to_label_set(text)

        if len(label_with_number) != int(number):
            print(fn,label_with_number)
            input()


import pickle
from copy import deepcopy
def SOA(text, label):
    '''
    input sample:
        text:   "A page with 1 Background Image, 1 Text, 1 Text Button, 1 Image, 1 Text Button, 1 Icon and 2 Text Button. "
        label:  [11, 1, 5, 2, 5, 3, 5, 5]
    '''

    matched_number = 0

    _, label_with_number = parsing_text_to_label_set(text)
   
    nl_labels = deepcopy(label_with_number)
    ui_labels = deepcopy(label)

    for i in range(len(ui_labels)):
        for j in ui_labels:
            if j in nl_labels:
                ui_labels.remove(j)
                nl_labels.remove(j)
                matched_number += 1

    return matched_number

def _check_nl_layout_semantic_align():

    with open('fid_data/RICO_10_filtered.pkl','rb')as f:
        layout = pickle.load(f)
    
    with open('../data/dataset/layout_nl_10.txt','r', encoding='utf-8') as f:
        text = f.read().rstrip('\n').split('\n')
    text_dataset = {}
    text_file_list = []
    for item in text:
        fn = item.split('\t')[0]
        text_dataset[fn] = [str(item.split('\t')[1])]
        text_file_list.append(fn)

    output = []
    for item in layout:
        if str(item['fn']) in text_file_list:
            item['text'] = text_dataset[str(item['fn'])]
            output.append(item)
    
    total_layout = len(output)
    correct_layout = 0

    total_ele = 0
    correct_ele = 0
    wrong = 0

    for l in output:

        text = l['text'][0]
        label = l['label']

        matched_number = SOA(text, label)
        
        # print(label,ori_label,matched_number)
        # input()

        if matched_number == len(label):
            correct_layout += 1
        else:
            wrong += 1
            print(l['fn'], text, label)
            input()

        total_ele += len(label)
        correct_ele += matched_number
    
    SOA_layout = correct_layout / total_layout
    SOA_ele = correct_ele / total_ele

    return SOA_layout, SOA_ele, wrong

# SOA_layout, SOA_ele, wrong = _check_nl_layout_semantic_align()
# print(SOA_layout, SOA_ele, wrong)