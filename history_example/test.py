# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/4/28 10:04'

import matplotlib.pyplot as plt
import time
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# define placeholder for inputs to network
with tf.name_scope('inputsdong'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_inputdong')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_inputdong')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)
# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
with tf.name_scope('traindong'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)
# important step
sess.run(tf.global_variables_initializer())

# tensorboard --logdir=logs