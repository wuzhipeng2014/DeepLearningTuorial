# encoding:utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from DLGetStarting.ImprovingDeepNuralNetwork.week3.tensorflow.ALUtils import *
from DLGetStarting.ImprovingDeepNuralNetwork.week3.tensorflow.tf_utils import load_dataset, random_mini_batches, \
    convert_to_one_hot, predict

np.random.seed(1)

## 计算损失函数

y_hat = tf.constant(36, name='y_hat')  # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')  # Define y. Set to 39

loss1 = tf.Variable((y - y_hat) ** 2, name='loss')  # Create a variable for the loss

init = tf.global_variables_initializer()  # When init is run later (session.run(init)),
# the loss variable will be initialized and ready to be computed
with tf.Session() as session:  # Create a session and print the output
    session.run(init)  # Initializes the variables
    print(session.run(loss1))  # Prints the loss

sess=tf.Session()
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)


print(sess.run(c))


# Change the value of x in the feed_dict

x = tf.placeholder(tf.int64, name = 'x')
print(sess.run(2 * x, feed_dict = {x: 5}))
sess.close()

## 线性函数
print( "result = " + str(linear_function()))


## sigmod函数

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))


## 逻辑回归损失函数
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))


## one-hot编码
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))

## 生成ones矩阵

print ("ones = " + str(ones([3])))



