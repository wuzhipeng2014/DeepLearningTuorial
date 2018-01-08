#encoding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from DLGetStarting.NeuralNetWorkAndDeepLearning.week2.testCases import *
from DLGetStarting.NeuralNetWorkAndDeepLearning.week2.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import sklearn
import sklearn.datasets
import sklearn.linear_model

print("test")

np.random.seed(1)

X,Y=load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

plt.show()

