import os
import errno

RICO_W =1.440
RICO_H =2.560
RICO_MAX_LABEL_NUM = 25

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