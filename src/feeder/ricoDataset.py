import os
import pickle
import random
import torch
import numpy as np
import math
from torch.utils.data import Dataset



class RicoDataset(Dataset):
    '''
    dataset Loader for rico
    '''
    def __init__(self, 
                data_path,
                debug=False
                ) -> None:
        super().__init__()
        self.data_path = data_path
        self.debug = debug
        self.data = []
        max_len = -1

        ''' load data from file'''
        with open(self.data_path, 'rb+') as f:
            # data_temp = pickle.load(f)
            # max_len = -1
            # for layout in data_temp:
            #     batch_data = []
            #     max_len = max(max_len, len(layout['label']) * 5)
            #     for i in range(len(layout['label'])):
            #         batch_data.append((layout['label'][i]))
            #         batch_data.append((math.ceil(layout['box'][i][0] * 127)))              # x1
            #         batch_data.append((math.ceil(layout['box'][i][1] * 127)))              # y1
            #         batch_data.append((math.ceil((layout['box'][i][2]-  layout['box'][i][0]) * 127)))   # w
            #         batch_data.append((math.ceil((layout['box'][i][3]-  layout['box'][i][1]) * 127)))   # h 
            #     self.data.append(batch_data)

            # # padding layout
            # for i in range(len(self.data)):
            #     if len(self.data[i]) < max_len:
            #         for j in range(max_len - len(self.data[i])):
            #             self.data[i].append(-1)

            # self.data = np.array(self.data)

            data_temp = pickle.load(f)
            for layout in data_temp:
                batch_data = ''
                for i in range(len(layout['label'])):
                    batch_data = batch_data + ' ' + str((layout['label'][i])) + ' ' + \
                                    str(math.ceil(layout['box'][i][0] * 127)) + ' ' + str(math.ceil(layout['box'][i][1] * 127)) + ' ' + \
                                    str(math.ceil((layout['box'][i][2]-  layout['box'][i][0]) * 127)) + ' ' + \
                                    str(math.ceil((layout['box'][i][3]-  layout['box'][i][1]) * 127))
                self.data.append(batch_data.lstrip().rstrip())

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    
    data_loader = RicoDataset(data_path='c:/Users/lenovo/Documents/Code/GitHub/BLT/dataset/RICO.pkl')
    print(data_loader.__getitem__(0))
