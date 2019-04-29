# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/4/29 22:56'

import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
A_2 = np.array([[1, 2, 3], [4, 5, 6]])

B = [[1, 2], [3, 4], [5, 6]]

C = np.dot(A, B)

D = np.array([2, 4, 6])
D_2 = np.array([1, 3, 5])

print(np.dot(D, D_2))

# print(C)
# print(C.shape)
# print(C.ndim)

# print(D.shape)
# (3,)
# print(D.ndim)
# 1

# print(np.dot(A, D))
# [28, 64]

# E = np.array([[2, 4]])

# print(E.shape)
# print(E.ndim)
# (1, 2)
# 2

# F = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#
# print(F.shape)
# print(F.ndim)
#
# print(np.dot(E, F))
# # [[22, 28, 34, 40]

print(A+A_2)
print((A+A_2).shape)