import json
import pickle
import matplotlib.pyplot as plt
from utils import *
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
    with open('../output/{}.json'.format(fn),'r')as f:
        layout_list = json.load(f)

    for layout in layout_list[:MAX_SHOW_NUM]:
        fig = plt.figure()
        ax1 = fig.add_subplot(121, aspect='equal')
        ax1.title.set_text('original')
        ax2 = fig.add_subplot(122, aspect='equal')
        ax2.title.set_text('generation')

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


if __name__ == '__main__':

    # fn = 'stage2/longseq/20/train_rec'
    # dataset = 'RICO_10'
    # reconstruction(fn, dataset)

    dataset = 'RICO_20'
    fn = 'stage2/20/test_gen'
    generation(fn, dataset)

    dataset = 'RICO_10'
    fn = 'stage2/10/test_gen'
    generation(fn, dataset)

    #dataset = 'RICO_10'
    #ori_dataset(dataset)
