import json
import glob
from tqdm import tqdm
import pickle

MAX_CLASS_NUM = 25
MAX_ELE_NUM = 20

def correct_bounds(bounds):
    left = min(bounds[0],bounds[2])
    top = min(bounds[1],bounds[3])
    right = max(bounds[0],bounds[2])
    bottom = max(bounds[1],bounds[3])

    return [left, top, right, bottom]

def group_box_list(group, box, structure):

    if 'componentLabel' in group:
        bounds = correct_bounds(group['bounds'])
        b = [bounds,group['componentLabel']]
        box.append(b)
        structure.append(len(box)-1)

    if 'children' in group:
        if group['children'] != []:
            for g in group['children']:
                box, s= group_box_list(g, box, [])
                structure.append(s)
        else:
            if 'componentLabel' in group:
                bounds = correct_bounds(group['bounds'])
                b = [bounds,group['componentLabel']]
                box.append(b)
                structure.append(len(box)-1)
    
    return box, structure

def normalization(box_pos,w,h):
    pos = [0.0,0.0,0.0,0.0]
    pos[0] = min(max(box_pos[0] / w, 0.0), 1.0)
    pos[2] = min(max(box_pos[2] / w, 0.0), 1.0)
    pos[1] = min(max(box_pos[1] / h, 0.0), 1.0)
    pos[3] = min(max(box_pos[3] / h, 0.0), 1.0)

    if pos[2]<pos[0]:
        t = pos[2]
        pos[2] = pos[0]
        pos[0] = t

    if pos[3]<pos[1]:
        t = pos[3]
        pos[3] = pos[1]
        pos[1] = t
    
    # if pos == [1.0, 0.11015625, 1.0, 0.43828125] or pos==[1.0, 0.327734375, 1.0, 0.340234375]:
    #     print(box_pos)
    #     input()
    
    return pos

def filter_this_layout(layout, number, classes):
    '''
        filter out the data by:
            element number 
            element class
    '''

    if (len(layout)>number)or(len(layout)<1):
        return True
    for i in range(len(layout)):
        if layout[i][1] not in classes:
            return True
    
    for ele in layout:
        box = ele[0]
        if (box[0]==box[2])or(box[1]==box[3])or(box[0]<0.0)or(box[1]<0.0)or(box[2]>1440.0)or(box[3]>2560.0):
            return True
    
    return False

def filter_same_overlap_ele(layout):
    
    filtered_box = []
    filtered_label = []

    for i in range(len(layout['box'])):
        box = layout['box'][i]
        label = layout['label'][i]
        if (box in filtered_box):
            index = filtered_box.index(box)
            del filtered_box[index]
            del filtered_label[index]
        filtered_box.append(box)
        filtered_label.append(label)

    return {'box':filtered_box,'label':filtered_label, 'fn':layout['fn']}


if __name__ == "__main__":
    flist = glob.glob(rf'../../data/RICO/semantic-annotation/*.json')
    flist.sort(key=lambda x: int(x.split('\\')[-1][:-5]))
    flist = flist
    print(len(flist))

    sorted_classes =  ['Text', 'Image', 'Icon', 'List Item', 'Text Button', 'Toolbar', 'Web View', 'Input', 'Card', 'Advertisement', 'Background Image', 'Drawer', 'Radio Button', 'Checkbox', 'Multi-Tab', 'Pager Indicator', 'Modal', 'On/Off Switch', 'Slider', 'Map View', 'Button Bar', 'Video', 'Bottom Navigation', 'Number Stepper', 'Date Picker']

    rico_dataset = []


    for f in tqdm(flist):

        box = []
        structure = []
        android_tree = json.load(open(f,'rb'))

        assert 'children' in android_tree
        
        sorted_bbox, group_tree = group_box_list(android_tree, box, structure)
        
        if len(sorted_bbox) == 0:
            continue

        if filter_this_layout(sorted_bbox, MAX_ELE_NUM, sorted_classes[:MAX_CLASS_NUM]):
            continue
        
        l = {}
        l["fn"] = int(f.split('\\')[-1].split('.')[0])

        label = []
        box = []

        for i in range(len(sorted_bbox)):  
            label.append(sorted_classes.index(sorted_bbox[i][1])+1)
            box.append(normalization(sorted_bbox[i][0],1440,2560))

        l["label"] = label
        l["box"] = box
        
        l = filter_same_overlap_ele(l)

        rico_dataset.append(l)

    print(len(rico_dataset)) 

    with open('../dataset/RICO.pkl','wb')as f:
        pickle.dump(rico_dataset,f)
    # with open('./dataset/RICO.json','w')as f:
    #     json.dump(rico_dataset,f)

    

