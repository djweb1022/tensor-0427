# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/4/27 14:34'

import tensorflow as tf

tf.enable_eager_execution()

a = tf.constant(1)
b = tf.constant(1)
c = tf.add(a, b)    # 也可以直接写 c = a + b，两者等价

print(c)

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)

print(C)
