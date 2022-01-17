from copy import deepcopy
from math import log
import numpy as np
import json

import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from collections import OrderedDict

from utils import FidNet
from utils import parsing_text_to_label_set

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

    return matched_number, label_with_number

def SOA_by_ori_label(ori_label, pred_label):
    matched_number = 0
    nl_labels = deepcopy(ori_label)
    ui_labels = deepcopy(pred_label)
    for i in range(len(ui_labels)):
        for j in ui_labels:
            if j in nl_labels:
                ui_labels.remove(j)
                nl_labels.remove(j)
                matched_number += 1

    return matched_number

def alignment(boxes):

    '''
    input sample:
        [[x1,y1,x2,y2],...]
    '''
    align_error = 0.0

    if len(boxes) < 2:
        return align_error

    for i in range(len(boxes)):
        l,xc,r,t,yc,b = [],[],[],[],[],[]
        for j in range(len(boxes)):
            if i == j:
                continue
            l.append(max(min(abs(boxes[i][0] - boxes[j][0]),1.0),0.0))
            r.append(max(min(abs(boxes[i][2] - boxes[j][2]),1.0),0.0))
            t.append(max(min(abs(boxes[i][1] - boxes[j][1]),1.0),0.0))
            b.append(max(min(abs(boxes[i][3] - boxes[j][3]),1.0),0.0))
            xc.append(max(min(abs((boxes[i][2]+boxes[i][0])/2.0-(boxes[j][2]+boxes[j][0])/2.0),1.0),0.0))
            yc.append(max(min(abs((boxes[i][3]+boxes[i][1])/2.0-(boxes[j][3]+boxes[j][1])/2.0),1.0),0.0))
        
        l,xc,r,t,yc,b = -np.log(1-min(l)), -np.log(1-min(xc)), -np.log(1-min(r)), -np.log(1-min(t)), -np.log(1-min(yc)), -np.log(1-min(b))
        align_error += min(l,xc,r,t,yc,b)
    
    return align_error

def overlap(boxes):
    '''
    input sample:
        [[x1,y1,x2,y2],...]
    '''
    overlapping = 0.0

    for i in range(len(boxes)):
        x1 = min(boxes[i][0],boxes[i][2]) 
        x2 = max(boxes[i][0],boxes[i][2]) 
        y1 = min(boxes[i][1],boxes[i][3]) 
        y2 = max(boxes[i][1],boxes[i][3])
        s_i = (x2-x1) * (y2-y1)

        for j in range(len(boxes)):
            if i == j:
                continue
            x3 = min(boxes[j][0],boxes[j][2]) 
            x4 = max(boxes[j][0],boxes[j][2]) 
            y3 = min(boxes[j][1],boxes[j][3]) 
            y4 = max(boxes[j][1],boxes[j][3]) 

            w = min(x2,x4) - max(x1,x3)
            h = min(y2,y4) - max(y1,y3)

            if w<0 or h<0:
                continue
            else:
                if s_i > 0:
                    s_ij = w * h
                    overlapping += s_ij / s_i

    return overlapping

def IoU(boxes):
    '''
    mean IoU
    input sample:
        [[x1,y1,x2,y2],...]
    '''

    iou = 0.0

    for i in range(len(boxes)):
        x1 = min(boxes[i][0],boxes[i][2])
        x2 = max(boxes[i][0],boxes[i][2])
        y1 = min(boxes[i][1],boxes[i][3])
        y2 = max(boxes[i][1],boxes[i][3])

        for j in range(len(boxes)):

            if i == j:
                continue

            x3 = min(boxes[j][0],boxes[j][2])
            x4 = max(boxes[j][0],boxes[j][2])
            y3 = min(boxes[j][1],boxes[j][3])
            y4 = max(boxes[j][1],boxes[j][3])

            w = min(x2,x4) - max(x1,x3)
            h = min(y2,y4) - max(y1,y3)

            if w<0 or h<0:
                continue
            else:
                S1 = (x2-x1)*(y2-y1)
                S2 = (x4-x3)*(y4-y3)
                S_cross = w * h
                iou += S_cross/(S1+S2-S_cross + 1e-10)

    return iou / len(boxes)

class LayoutFID():
    def __init__(self, cfg, device='cpu'):
        
        self.cfg = cfg
        self.model = FidNet(cfg).to(device)

        # load pre-trained LayoutNet
        fid_model_dir = 'fid_data/net_paras_fid.pth'
        state_dict_load = torch.load(fid_model_dir,map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()                        
        for k, v in state_dict_load.items():                  
            namekey = k[7:] if k.startswith('module.') else k 
            new_state_dict[namekey] = v                       
        self.model.load_state_dict(new_state_dict)

        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        # if type(self.real_features) == list:
        #     feats_2 = np.concatenate(self.real_features)
        #     self.real_features = feats_2
        # else:
        #     feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)

        # mu_2 = np.mean(feats_2, axis=0)
        # sigma_2 = np.cov(feats_2, rowvar=False)

        with open('./fid_data/real_mu_sigma.json','r')as f:
            log = json.load(f)
        
        mu_2 = np.array(log[self.cfg.dataset.split('_')[-1]]['mu'])
        sigma_2 = np.array(log[self.cfg.dataset.split('_')[-1]]['sigma'])

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)





        
        







