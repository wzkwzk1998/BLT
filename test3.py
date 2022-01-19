import random

# l = [i for i in range(100)]


# train_ids = random.sample(l, 85)
# test_and_val = list(set(l).difference(set(train_ids)))
# print(test_and_val)
# test = random

a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]

b = [[[13,14,15],[16,17,18],[22,23,24]],[[19,20,21]]]

c = a + b
# print(c)
import json
import pickle

# test_dict = {'generation':c}
# with open('./test_json.json' ,'w') as f:
#     json.dump(test_dict, f)


# with open('./out/2022-01-19-10-51.json' ,'r') as f:
#     c = json.load(f)
#     print(len(c['generation']))
#     print(len(c['origin']))


with open('./data/RICO_test.pkl', 'rb+') as f:
    data = pickle.load(f)
    print('test data {}'.format(len(data)))

with open('./data/RICO_train.pkl', 'rb+') as f:
    data = pickle.load(f)
    print('train data {}'.format(len(data)))

with open('./data/RICO_val.pkl', 'rb+') as f:
    data = pickle.load(f)
    print('val data {}'.format(data))





