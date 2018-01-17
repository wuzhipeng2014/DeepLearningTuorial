# encoding:utf-8

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from DLGetStarting.ImprovingDeepNuralNetwork.week3.tensorflow.ALUtils import *
from DLGetStarting.ImprovingDeepNuralNetwork.week3.tensorflow.improv_utils import *
from DLGetStarting.ImprovingDeepNuralNetwork.week3.tensorflow.tf_utils import *

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



## 手势识别
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
plt.show()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

## 创建placeholders
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))

## 参数初始化
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

## 前向传播
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


## 计算损失函数

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))

## 测试模型
parameters = model(X_train, Y_train, X_test, Y_test)


## 测试其他图片
import scipy
from PIL import Image
from scipy import ndimage

## START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "thumbs_up_org.jpg"
my_image = "thumbs_up2.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))


