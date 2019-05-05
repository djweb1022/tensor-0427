# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/5 12:16'

import numpy as np
from keras.utils import to_categorical


# 获得一个列表，对列表中非重复元素添加索引编号，返回索引列表
def get_index(data):
    token_index = {}
    for i, word in enumerate(data):
        if word not in token_index:
            token_index[word] = len(token_index) + 1

    num_list = np.array([])
    for word in data:
        num_list = np.append(num_list, token_index[word])

    return num_list


# 获得索引列表，进行one-hot编码，返回编码列表
def encode(num_list):
    print('Shape of data (BEFORE encode): %s' % str(num_list.shape))
    encoded = to_categorical(num_list)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded


data = np.array([1, 5, 3, 8, 20, 10, 12, 5, 8])
print(data)
print(data.shape)


data_num = get_index(data)
encoded_data = encode(data_num)
print(encoded_data)


data2 = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = np.array(data2)
print(values)
# ['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']

data2_index = get_index(data2)

encoded_data = encode(data2_index)
print(encoded_data)

