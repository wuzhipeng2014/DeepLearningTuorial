# encoding:utf-8
from DLGetStarting.ImprovingDeepNuralNetwork.week1.regularization.reg_utils import load_2D_dataset
import matplotlib.pyplot as plt
from DLGetStarting.ImprovingDeepNuralNetwork.week1.regularization.DNNRegularizationUtils import *
from DLGetStarting.ImprovingDeepNuralNetwork.week1.regularization.testCases import *

train_X, train_Y, test_X, test_Y = load_2D_dataset()


parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)


plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
plt.show()

## 添加L2正则项

A3, Y_assess, parameters = compute_cost_with_regularization_test_case()

print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))


X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

## 带L2正则项的反向转播
grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
print ("dW1 = "+ str(grads["dW1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("dW3 = "+ str(grads["dW3"]))

## 带正则项的模型训练
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

## 决策边界图
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

plt.show()








