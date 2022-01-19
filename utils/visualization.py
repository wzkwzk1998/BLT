import json
import pickle
import matplotlib.pyplot as plt
import os
import errno
from tqdm import tqdm
from utils import *

RICO_W = 1.440
RICO_H = 2.560
RICO_MAX_LABEL_NUM = 25
MAX_SHOW_NUM = 200

def ori_dataset(dataset):
    with open('../data/'+dataset+'.pkl','rb')as f:
        layout_list = pickle.load(f)

    for layout in layout_list[:MAX_SHOW_NUM]:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')
        ax1.title.set_text('original')
        
        if dataset.split('_')[0] == 'RICO':
            w = RICO_W
            h = RICO_H
            max_label_num = RICO_MAX_LABEL_NUM
            color = rico_color
            label_text = rico_label

        for i in range(len(layout['box'])):

            label = layout['label'][i]
            if label <= max_label_num:
                c = color(label)
                ls = label_text(label)
            else:
                c = 'black'
                ls = 'WRONG'
            box = layout['box'][i]
            ax1.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    facecolor=c, alpha=0.2
                )  
            )
            ax1.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    linewidth=1.5, edgecolor=c, fill=False
                )
            )
            ax1.text(box[0]*w+0.003, box[1]*h+0.015, ls, fontsize=6,color=c)
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)

        mkdir_if_missing('../output/{}/'.format(dataset))
        fig.savefig('../output/{}/{}'.format(dataset,layout['fn']), dpi=300, bbox_inches='tight')
        plt.close()


def reconstruction(fn, dataset):
    
    with open('../output/{}.json'.format(fn),'r')as f:
        layout_list = json.load(f)

    for layout in layout_list[:MAX_SHOW_NUM]:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, aspect='equal')
        ax1.title.set_text('original')
        ax2 = fig.add_subplot(122, aspect='equal')
        ax2.title.set_text('reconstruction')

        #plt.figtext(0.0,0.0,layout['text'][0])
        
        if dataset.split('_')[0] == 'RICO':
            w = RICO_W
            h = RICO_H
            max_label_num = RICO_MAX_LABEL_NUM
            color = rico_color
            label_text = rico_label

        for i in range(len(layout['ori_box'])):
        
            label = layout['ori_label'][i]
            
            if label <= max_label_num:
                c = color(label)
                ls = label_text(label)
            else:
                c = 'black'
                ls = 'WRONG'
            box = layout['ori_box'][i]
            ax1.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    facecolor=c, alpha=0.2
                )  
            )
            ax1.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    linewidth=1.5, edgecolor=c, fill=False
                )
            )
            ax1.text(box[0]*w+0.003, box[1]*h+0.015, ls, fontsize=6,color=c)

        for i in range(len(layout['pred_box'])):

            label = layout['pred_label'][i]
            if label <= max_label_num:
                c = color(label)
                ls = label_text(label)
            else:
                c = 'black'
                ls = 'WRONG'
            box = layout['pred_box'][i]
            ax2.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    facecolor=c, alpha=0.2
                )  
            )
            ax2.add_patch(
                plt.Rectangle(
                    (box[0]*w, box[1]*h),  # (x,y)矩形左下角
                    box[2]*w-box[0]*w,  # width长
                    box[3]*h-box[1]*h,  # height宽
                    linewidth=1.5, edgecolor=c, fill=False
                )
            )
            ax2.text(box[0]*w+0.003, box[1]*h+0.015, ls, fontsize=6,color=c)
        
        
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)
        
        mkdir_if_missing('../output/{}/'.format(fn))
        fig.savefig('../output/{}/{}'.format(fn,layout['fn']), dpi=300, bbox_inches='tight')
        plt.close()


def generation(fn, dataset):

    with open('./out/{}.json'.format(fn), 'r') as f:
        layout_list = json.load(f)

    for idx, (gen_layout, orig_layout) in tqdm(enumerate(zip(layout_list['generation'][:MAX_SHOW_NUM], layout_list['origin'][:MAX_SHOW_NUM]))):
        fig = plt.figure()
        ax1 = fig.add_subplot(121, aspect='equal')
        ax1.title.set_text('original')
        ax2 = fig.add_subplot(122, aspect='equal')
        ax2.title.set_text('generation')

        if dataset.split('_')[0] == 'RICO':
            w = RICO_W
            h = RICO_H
            max_label_num = RICO_MAX_LABEL_NUM
            color = rico_color
            label_text = rico_label

        for box in  orig_layout:
            label_id = box[0]
            if label_id <= max_label_num:
                c = color(label_id)
                label = label_text(label_id)           # map id -> text
            else:
                c = 'black'
                label = 'WRONG'
            # draw
            ax1.add_patch(
                plt.Rectangle(
                    ((box[1] / 127.0) * w, (box[2] / 127.0) * h),  # (x,y)矩形左下角
                    (box[3] / 127.0) * w,  # width长
                    (box[4] / 127.0) * h,  # height宽
                    facecolor=c, alpha=0.2
                ) 
            )
            ax1.add_patch(
                plt.Rectangle(
                    ((box[1] / 127.0) * w, (box[2] / 127.0) * h),  # (x,y)矩形左下角
                    (box[3] / 127.0) * w,  # width长
                    (box[4] / 127.0) * h,  # height宽
                    linewidth=1.5, edgecolor=c, fill=False
                )
            )
            ax1.text(box[1] / 127.0 * w + 0.003, box[2] / 127.0 * h + 0.015, label, fontsize=6,color=c)
        
        for box in gen_layout:
            label_id = box[0]
            if label_id <= max_label_num:
                c = color(label_id)
                label = label_text(label_id)           # map id -> text
            else:
                c = 'black'
                label = 'WRONG'
            # draw
            ax2.add_patch(
                plt.Rectangle(
                    ((box[1] / 127.0) * w, (box[2] / 127.0) * h),  # (x,y)矩形左下角
                    (box[3] / 127.0) * w,  # width长
                    (box[4] / 127.0) * h,  # height宽
                    facecolor=c, alpha=0.2
                ) 
            )
            ax2.add_patch(
                plt.Rectangle(
                    ((box[1] / 127.0) * w, (box[2] / 127.0) * h),  # (x,y)矩形左下角
                    (box[3] / 127.0) * w,  # width长
                    (box[4] / 127.0) * h,  # height宽
                    linewidth=1.5, edgecolor=c, fill=False
                )
            )
            ax2.text(box[1] / 127.0 * w + 0.003, box[2] / 127.0 * h + 0.015, label, fontsize=6,color=c)

        
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax2.set_xlim(0, w)
        ax2.set_ylim(h, 0)

        mkdir_if_missing('./picture/{}/'.format(fn))
        fig.savefig('./picture/{}/{}'.format(fn, idx), dpi=300, bbox_inches='tight')
        plt.close()


# def generation(fn, dataset):
#     with open('../output/{}.json'.format(fn),'r')as f:
#         layout_list = json.load(f)

#     for layout in layout_list[:MAX_SHOW_NUM]:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(121, aspect='equal')
#         ax1.title.set_text('original')
#         ax2 = fig.add_subplot(122, aspect='equal')
#         ax2.title.set_text('generation')

#         #plt.figtext(0.0,0.0,layout['text'][0])
        
#         if dataset.split('_')[0] == 'RICO':
#             w = RICO_W
#             h = RICO_H
#             max_label_num = RICO_MAX_LABEL_NUM
#             color = rico_color
#             label_text = rico_label

#         for i in range(len(layout['ori_box'])):

#             label = layout['ori_label'][i]
#             if label <= max_label_num:
#                 c = color(label)
#                 ls = label_text(label)
#             else:
#                 c = 'black'
#                 ls = 'WRONG'
#             box = layout['ori_box'][i]
#             ax1.add_patch(
#                 plt.Rectangle(
#                     (box[0]*w, box[1]*h),  # (x,y)矩形左下角
#                     box[2]*w-box[0]*w,  # width长
#                     box[3]*h-box[1]*h,  # height宽
#                     facecolor=c, alpha=0.2
#                 )  
#             )
#             ax1.add_patch(
#                 plt.Rectangle(
#                     (box[0]*w, box[1]*h),  # (x,y)矩形左下角
#                     box[2]*w-box[0]*w,  # width长
#                     box[3]*h-box[1]*h,  # height宽
#                     linewidth=1.5, edgecolor=c, fill=False
#                 )
#             )
#             ax1.text(box[0]*w+0.003, box[1]*h+0.015, ls, fontsize=6,color=c)

#         for i in range(len(layout['pred_box'])):
#             label = layout['pred_label'][i]
#             if label <= max_label_num:
#                 c = color(label)
#                 ls = label_text(label)
#             else:
#                 c = 'black'
#                 ls = 'WRONG'
#             box = layout['pred_box'][i]
#             ax2.add_patch(
#                 plt.Rectangle(
#                     (box[0]*w, box[1]*h),  # (x,y)矩形左下角
#                     box[2]*w-box[0]*w,  # width长
#                     box[3]*h-box[1]*h,  # height宽
#                     facecolor=c, alpha=0.2
#                 )  
#             )
#             ax2.add_patch(
#                 plt.Rectangle(
#                     (box[0]*w, box[1]*h),  # (x,y)矩形左下角
#                     box[2]*w-box[0]*w,  # width长
#                     box[3]*h-box[1]*h,  # height宽
#                     linewidth=1.5, edgecolor=c, fill=False
#                 )
#             )
#             ax2.text(box[0]*w+0.003, box[1]*h+0.015, ls, fontsize=6,color=c)
        
#         ax1.set_xlim(0, w)
#         ax1.set_ylim(h, 0)
#         ax2.set_xlim(0, w)
#         ax2.set_ylim(h, 0)

#         mkdir_if_missing('../output/{}/'.format(fn))
#         fig.savefig('../output/{}/{}'.format(fn,layout['fn']), dpi=300, bbox_inches='tight')
#         plt.close()


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise 

def rico_color(label):
    vocab = {
        0: 'black',
        1: 'orange',
        2: 'red',
        3: 'tan',
        4: 'silver',
        5: 'yellow',
        6: 'green',
        7: 'blue',
        8: 'violet',
        9: 'c',
        10:'goldenrod',
        11:'teal',
        12:'teal',
        13:'pink',
        14: 'black',
        15: 'silver',
        16: 'red',
        17: 'tan',
        18: 'orange',
        19: 'blue',
        20: 'green',
        21: 'blue',
        22: 'violet',
        23: 'c',
        24:'goldenrod',
        25:'teal'
    }
    return vocab[label]

def rico_label(label):
    vocab = {
        0: 'Nonetype', # for pad
        1:'Text', 
        2:'Image', 
        3:'Icon', 
        4:'List Item', 
        5:'Text Button', 
        6:'Toolbar', 
        7:'Web View', 
        8:'Input',
        9:'Card',
        10:'Advertisement', 
        11:'Background Image', 
        12:'Drawer', 
        13:'Radio Button', 
        14:'Checkbox', 
        15:'Multi-Tab', 
        16:'Pager Indicator', 
        17:'Modal', 
        18:'On/Off Switch', 
        19:'Slider', 
        20:'Map View', 
        21:'Button Bar', 
        22:'Video', 
        23:'Bottom Navigation', 
        24:'Number Stepper', 
        25:'Date Picker'
    }

    return vocab[label]


if __name__ == '__main__':

    # fn = 'stage2/longseq/20/train_rec'
    # dataset = 'RICO_10'
    # reconstruction(fn, dataset)

    dataset = 'RICO_20'
    fn = '2022-01-19-14-29'
    generation(fn, dataset)

    # dataset = 'RICO_10'
    # fn = 'blt/'
    # generation(fn, dataset)

    #dataset = 'RICO_10'
    #ori_dataset(dataset)
