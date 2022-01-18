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

# test_dict = {'generation':c}
# with open('./test_json.json' ,'w') as f:
#     json.dump(test_dict, f)


with open('./test_json.json' ,'r') as f:
    c = json.load(f)
    print(c['generation'])




