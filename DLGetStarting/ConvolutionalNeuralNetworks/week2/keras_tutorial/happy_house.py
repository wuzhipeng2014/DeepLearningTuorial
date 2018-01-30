# encoding:utf-8

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K

from DLGetStarting.ConvolutionalNeuralNetworks.week2.keras_tutorial.kt_utils import *
from DLGetStarting.ConvolutionalNeuralNetworks.week2.keras_tutorial.happy_house_utils import *

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


happyModel = HappyModel((64,64,3))

happyModel.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

happyModel.fit(x=X_train, y=Y_train, epochs=30, batch_size=20)


preds = happyModel.evaluate(x=X_test, y=Y_test,)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


## 在自己的图片上测试

# for i in range(1,5):
#     img_path = 'images/'+str(i) + '.jpg'
#     ### END CODE HERE ###
#     img = image.load_img(img_path, target_size=(64, 64))
#     imshow(img)
#
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#
#     print(happyModel.predict(x))





