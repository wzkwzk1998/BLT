import json
import pickle
from metric import SOA, SOA_by_ori_label, alignment, overlap, IoU, LayoutFID
from utils import LayoutDataset
from fid_data import config

import torch
from torch.utils.data import DataLoader

FID_BATCH_SIZE = 100

def cal_SOA(output):

    total_layout = len(output)
    correct_layout = 0

    total_ele = 0
    correct_ele = 0

    for l in output:

        text = l['text']
        label = l['pred_label']
        #ori_label = l['ori_label']

        matched_number, nl_label = SOA(text, label)
        #matched_number = SOA_by_ori_label(ori_label, label)
        
        # print(label,ori_label,nl_label)
        # input()

        if matched_number == len(nl_label):
            correct_layout += 1

        total_ele += len(nl_label)
        correct_ele += matched_number
    
    SOA_layout = correct_layout / total_layout
    SOA_ele = correct_ele / total_ele

    return SOA_layout, SOA_ele

def cal_alignment(output):

    align_error = 0.0

    for l in output:
        boxes = l['pred_box']
        align_error += alignment(boxes)
    
    return align_error / len(output)

def cal_overlap(output):

    overlapping = 0.0

    for l in output:
        boxes = l['pred_box']
        overlapping += overlap(boxes)
    
    return overlapping / len(output)

def cal_IoU(output):

    iou = 0.0

    for l in output:
        boxes = l['pred_box']
        iou += IoU(boxes)
    
    return iou / len(output)

def cal_FID(cfg, dataset, output):
    batch_size = FID_BATCH_SIZE

    device = torch.device("cuda:0,1,2,3" if torch.cuda.is_available() else "cpu")

    #real_dataset = LayoutDataset(cfg, dataset, real=True)
    fake_dataset = LayoutDataset(cfg, output)

    #real_data_loader = DataLoader(dataset=real_dataset, batch_size=batch_size, shuffle=False)
    fake_data_loader = DataLoader(dataset=fake_dataset, batch_size=batch_size, shuffle=False)

    fid = LayoutFID(cfg, device)

    # for data in real_data_loader:
        
    #     real_data = data      
    #     real_data = {key: real_data[key].to(device) for key in real_data} 
    #     real_bbox = real_data['box']
    #     real_label = real_data['label']
    #     real_mask  = (real_data['label_mask'] == 0).cumsum(dim=0) > 0
    #     fid.collect_features(real_bbox, real_label, real_mask, real=True)

    for data in fake_data_loader:

        fake_data = data
        fake_data = {key: fake_data[key].to(device) for key in fake_data} 
        fake_bbox = fake_data['box']
        fake_label = fake_data['label']
        fake_mask  = (fake_data['label_mask'] == 0).cumsum(dim=0) > 0
        #fid.collect_features(real_bbox, real_label, real_mask)
        fid.collect_features(fake_bbox, fake_label, fake_mask)
    
    fid_score = fid.compute_score()

    return fid_score



if __name__ == '__main__':

    with open('../output/stage2_gpt/10_longseq/test_gen.json','r')as f:
        output = json.load(f)

    # SOA
    SOA_layout, SOA_ele = cal_SOA(output)
    print('SOA_L: ', SOA_layout)
    print('SOA_E: ', SOA_ele)

    # alignment
    align_err = cal_alignment(output)
    print('Alignment:', align_err)

    # overlap
    overlapping = cal_overlap(output)
    print('Overlap:', overlapping)

    # IoU
    iou = cal_IoU(output)
    print('mIoU:', iou)

    # Fid
    cfg = config.parse_cfg()
    dataset_fn = 'fid_data/{}.pkl'.format(cfg.dataset)
    with open(dataset_fn,'rb')as f:
        dataset = pickle.load(f)
    fid = cal_FID(cfg, dataset, output)
    print('fid:', fid)
    
