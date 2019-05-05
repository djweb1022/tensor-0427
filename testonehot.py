# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/5 12:16'

import numpy as np
from keras.utils import to_categorical

def get_index(data):
    token_index = {}
    for i, word in enumerate(data):
        if word not in token_index:
            token_index[word] = len(token_index) + 1
    return token_index

data = np.array([1, 5, 3, 8, 20, 10])
# data = np.array(['aa', 'bb', 'cc', 'zz', 'cc', 'kk'])
print(data)
print(data.shape)
# [ 1  5  3  8 20 10]
# (6,)

data_index = get_index(data)

data_num = np.array([])
for num in data:
    data_num = np.append(data_num, data_index[num])

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded


encoded_data = encode(data_num)
print(encoded_data)
# Shape of data (BEFORE encode): (6,)
# Shape of data (AFTER  encode): (6, 21)
#
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

data2 = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = np.array(data2)
print(values)
# ['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']

data2_index = get_index(data2)
# {'cold': 1, 'warm': 2, 'hot': 3}

data2_num = np.array([])
for word in data2:
    data2_num = np.append(data2_num, data2_index[word])

print(data2_num)
# [1. 1. 2. 1. 3. 3. 2. 1. 2. 3.]
encoded_data = encode(data2_num)
print(encoded_data)
# Shape of data (BEFORE encode): (10,)
# Shape of data (AFTER  encode): (10, 4)
#
# [[0. 1. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 0. 1.]
#  [0. 0. 0. 1.]
#  [0. 0. 1. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]



