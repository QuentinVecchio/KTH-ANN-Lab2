# -*- coding: utf-8 -*-

import RBF
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def generateDataSet(N, V):
    X = list(np.random.multivariate_normal([V, V], [[1, 0], [0, 1]], N))
    X += list(np.random.multivariate_normal([-V, -V], [[1, 0], [0, 1]], N))
    T = [1] * N + [-1] * N
    p = np.random.permutation(len(X))
    return (np.array(X)[p]).T, np.array(T)[p]

def square(X):
    Y = np.empty(len(X))
    for i,x in enumerate(X):
        if x >= 0 and x < np.pi:
            Y[i] = 1
        else:
            Y[i] = -1
    return Y

def scenario3_1():
    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    NB_HIDDEN=10
    ##### Sinus Function #####
    YSinus = np.sin(2*X)
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = np.sin(2*XTest)
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    history = rbfSinus.fit(X=X, T=YSinus, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSinusRBF = rbfSinus.predict(XTest)
    graph.plotRBFInformations("Sinus function", XTest, YSinusTest, YSinusRBF, YSinus, history, [0, 2*np.pi, -1.5, 1.5])

    ##### Square Function #####
    YSquare = square(2*X)
    YSquare = YSquare.reshape((len(YSquare),1))
    YSquareTest = square(2*XTest)
    YSquareTest = YSquareTest.reshape((len(YSquareTest),1))
    rbfSquare = RBF.simpleRBF(lr=0.01, nb_eboch=10000, nb_hidden=NB_HIDDEN, batch_size=-1)
    #history = rbfSquare.fit(X=X, T=YSquare, mu=[0, 2*np.pi, np.pi-0.01, np.pi+0.01, np.pi-0.05, np.pi+0.05] , sigma=np.sqrt(1))
    history = rbfSquare.fit(X=X, T=YSquare, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSquareRBF = rbfSquare.predict(XTest)
    graph.plotRBFInformations("Square function", XTest, YSquareTest, YSquareRBF, YSquare, history, [0, 2*np.pi, -1.5, 1.5])

def scenario3_2():
    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    NB_HIDDEN=10
    ##### Sinus Function #####
    YSinus = np.sin(2*X)
    # Add some noise
    error = np.random.normal(0, 0.1, len(YSinus))
    YSinus += error
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = np.sin(2*XTest)
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    history = rbfSinus.fit(X=X, T=YSinus, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSinusRBF = rbfSinus.predict(XTest)
    graph.plotRBFInformations("Sinus function with noise", XTest, YSinusTest, YSinusRBF, YSinus, history, [0, 2*np.pi, -1.5, 1.5])

    ##### Square Function #####
    YSquare = square(2*X)
    # Add some noise
    error2 = np.random.normal(0, 0.1, len(YSquare))
    YSquare += error2
    YSquare = YSquare.reshape((len(YSquare),1))
    YSquareTest = square(2*XTest)
    YSquareTest = YSquareTest.reshape((len(YSquareTest),1))
    rbfSquare = RBF.simpleRBF(lr=0.01, nb_eboch=10000, nb_hidden=NB_HIDDEN, batch_size=1, neighbourhood_size=50)
    #history = rbfSquare.fit(X=X, T=YSquare, mu=[0, 2*np.pi, np.pi-0.01, np.pi+0.01, np.pi-0.05, np.pi+0.05] , sigma=np.sqrt(1))
    history = rbfSquare.fit(X=X, T=YSquare, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSquareRBF = rbfSquare.predict(XTest)
    graph.plotRBFInformations("Square function with noise", XTest, YSquareTest, YSquareRBF, YSquare, history, [0, 2*np.pi, -1.5, 1.5])

def scenario3_3():
    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi, num = N)
    NB_HIDDEN=10
    ##### Sinus Function #####

    YSinus = np.sin(2*X)
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = np.sin(2*XTest)
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    rbfSinus.initialisationMu(X)
    history = rbfSinus.fit(X=X, T=YSinus , sigma=np.sqrt(2*np.pi/NB_HIDDEN))#np.sqrt(2*np.pi/NB_HIDDEN)
    YSinusRBF = rbfSinus.predict(XTest)
    graph.plotRBFInformations("Sinus function", XTest, YSinusTest, YSinusRBF, YSinus, history, [0, 2*np.pi, -1.5, 1.5])

    ##### Square Function #####

    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    YSquare = square(2*X)
    YSquare = YSquare.reshape((len(YSquare),1))
    YSquareTest = square(2*XTest)
    YSquareTest = YSquareTest.reshape((len(YSquareTest),1))
    rbfSquare = RBF.simpleRBF(lr=0.2, nb_eboch=10000, nb_hidden=NB_HIDDEN, batch_size=1)
    rbfSquare.initialisationMu(X)
    #history = rbfSquare.fit(X=X, T=YSquare, mu=[0, 2*np.pi, np.pi-0.01, np.pi+0.01, np.pi-0.05, np.pi+0.05] , sigma=np.sqrt(1))
    history = rbfSquare.fit(X=X, T=YSquare , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSquareRBF = rbfSquare.predict(XTest)
    graph.plotRBFInformations("Square function", XTest, YSquareTest, YSquareRBF, YSquare, history, [0, 2*np.pi, -1.5, 1.5])
