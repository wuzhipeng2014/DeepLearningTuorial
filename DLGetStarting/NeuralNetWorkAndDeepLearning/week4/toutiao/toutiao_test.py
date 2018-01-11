# encoding:utf-8

# 使用神经网络预测
from sklearn import preprocessing
import numpy as np
## 将csv文件加载到数组中
from sklearn.externals.joblib.numpy_pickle_utils import xrange

from DLGetStarting.NeuralNetWorkAndDeepLearning.week4.imageClassification.l_layer_neural_network import *

def max_min_normalization(data_value, data_col_max_values, data_col_min_values):
    """ Data normalization using max value and min value

    Args:
        data_value: The data to be normalized
        data_col_max_values: The maximum value of data's columns
        data_col_min_values: The minimum value of data's columns
    """
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]

    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value[i][j] = \
                (data_value[i][j] - data_col_min_values[i]) / \
                (data_col_max_values[i] - data_col_min_values[i])

def nonlinearity_normalization_lg(data_value_after_lg, data_col_max_values_after_lg):
    """ Data normalization using lg

    Args:
        data_value_after_lg: The data to be normalized
        data_col_max_values_after_lg: The maximum value of data's columns
    """
    data_shape = data_value_after_lg.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value_after_lg[i][j] = data_value_after_lg[i][j] / data_col_max_values_after_lg[j]


def standard_deviation_normalization(data_value, data_col_means,
                                     data_col_standard_deviation):
    """ Data normalization using standard deviation

    Args:
        data_value: The data to be normalized
        data_col_means: The means of data's columns
        data_col_standard_deviation: The variance of data's columns
    """
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value[i][j] = \
                (data_value[i][j] - data_col_means[i]) / \
                data_col_standard_deviation[i]


train_matrix = np.loadtxt(open("/home/zhipengwu/secureCRT/train_toutiao_origin_feature_20180109.csv", "rb"),
                          delimiter=",", skiprows=0)
train_x_origin = train_matrix[:, 1:]
train_y = train_matrix[:, 0].reshape(1, -1)

train_x = train_x_origin.T

scaled_train_x = preprocessing.scale(train_x,axis=1)
train_x_max =train_x.max(axis=1)
train_x_min=train_x.min(axis=1)
train_x_mean = scaled_train_x.mean(axis=1)
train_x_std = scaled_train_x.std(axis=1)

test_matrix = np.loadtxt(open("/home/zhipengwu/secureCRT/test_toutiao_origin_feature_20180109.csv", "rb"),
                         delimiter=",", skiprows=0)
test_x_origin = test_matrix[:, 1:]
test_y = test_matrix[:, 0].reshape(1, -1)
test_x = test_x_origin.T

scaled_test_x = preprocessing.scale(test_x,axis=1)
test_x_max =test_x.max(axis=1)
test_x_min=test_x.min(axis=1)

test_x_mean = scaled_test_x.mean(axis=1)
test_x_std = scaled_test_x.std(axis=1)

print("train_x shape=" + str(train_x.shape))
print("train_y shape=" + str(train_y.shape))
print("train_x_mean shape=" + str(train_x_mean.shape))
print("train_x_std shape=" + str(train_x_std.shape))

# 特征标准化
max_min_normalization(train_x,train_x_max,train_x_min)
max_min_normalization(test_x,test_x_max,test_x_min)

# standard_deviation_normalization(train_x,train_x_mean,train_x_std)
# standard_deviation_normalization(test_x,test_x_mean,test_x_std)


layers_dims = [38,200,150,100,70,50,35, 28,22,15,10,5, 1]  # 5-layer model

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=30000, print_cost=True)

pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
