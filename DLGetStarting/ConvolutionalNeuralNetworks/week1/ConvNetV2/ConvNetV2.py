# encoding:utf-8
import numpy as np
import h5py
import matplotlib.pyplot as plt

from DLGetStarting.ConvolutionalNeuralNetworks.week1.ConvNetV2.ConveNetV2Utils import *

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

## 边界填充
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[1,:,:,1])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[1,:,:,1])

plt.show()

##