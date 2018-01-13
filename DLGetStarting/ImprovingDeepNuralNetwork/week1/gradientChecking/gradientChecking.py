# encoding:utf-8

from DLGetStarting.ImprovingDeepNuralNetwork.week1.gradientChecking.DNNUtils import *
from DLGetStarting.ImprovingDeepNuralNetwork.week1.gradientChecking.testCases import gradient_check_n_test_case

X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)


