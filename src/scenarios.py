# -*- coding: utf-8 -*-

import RBF
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def generateDataSet(N, V):
    X = list(np.random.multivariate_normal([V, V], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([-V, -V], [[1, 0], [0, 1]], N))
    T = [1] * N + [-1] * N
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.array(T)[p]


def scenario3_1():
    print("Not Implemented yet")

def scenario3_2():
    print("Not Implemented yet")

def scenario3_3():
    print("Not Implemented yet")
