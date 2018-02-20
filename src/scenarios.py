# -*- coding: utf-8 -*-

import RBF
import SOM
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
    rbfSquare = RBF.simpleRBF(lr=0.01, nb_eboch=10000, nb_hidden=NB_HIDDEN, batch_size=1)
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

def scenario4_1():
    # Read the animals data and create the 32x84 matrix
    fileAnimals = open("../dataset/animals.dat", "r")
    content = fileAnimals.read()
    fileAnimals.close()
    attributes = np.array(list(map(int, content.split(","))))
    attributes = attributes.reshape((32,84))
    # Read the animals names
    fileAnimalsName = open("../dataset/animalnames.txt", "r")
    content = fileAnimalsName.read()
    fileAnimalsName.close()
    names= [n.replace("\t","").replace("'","") for n in content.split("\n")]
    # Create the SOM
    som = SOM.SOM(lr=0.01, nb_eboch=20, nb_hidden=100, neighbourhood_size=50)
    print("Learning ...")
    som.fit(attributes, method="index")
    print("Predict ...")
    result = som.predict(attributes)
    result.sort(key=lambda tup: tup[1])
    for i,winner in result:
        print(names[i])

def scenario4_2():
    # Read the animals data and create the 32x84 matrix
    filecities = open("../dataset/cities.dat", "r")
    coordinates = []
    content = filecities.readlines()
    for line in content:
        if line[0] != '%' and len(line) > 1: # avoid comments and empty line
            line = line.split(';')[0]
            coordinates.append(np.array(list(map(float, line.split(",")))))

    filecities.close()
    coordinates = np.array(coordinates)
    coordinates = coordinates.reshape((10,2))
    graph.plotMap("Cities", coordinates)
    # Create the SOM
    som = SOM.SOM(lr=0.01, nb_eboch=50, nb_hidden=10, neighbourhood_size=2)
    print("Learning ...")
    som.fit(coordinates, method="index-circular")
    print("Predict ...")
    result = som.predict(coordinates)
    print(result)
    #
    # for i,winner in result:
    #     print(names[i])

def scenario4_3():
    fileName = open("../dataset/mpnames.txt", "r", encoding='iso-8859-1')
    MPnames = []
    content = fileName.readlines()
    for line in content:
        MPnames.append(line.replace('\n', ''))

    fileSex = open("../dataset/mpsex.dat", "r")
    MPsex=[]
    content = fileSex.readlines()
    for line in content:
        line = line.strip()
        if len(line) >= 1 and line[0] != '%': # avoid comments and empty line
            MPsex.append(line)

    fileParty = open("../dataset/mpparty.dat", "r")
    MPparty=[]
    content = fileParty.readlines()
    for line in content:
        line = line.strip()
        if len(line) >= 1 and line[0] != '%': # avoid comments and empty line
            MPparty.append(line)

    fileDistrict = open("../dataset/mpdistrict.dat", "r")
    MPdistrict=[]
    content = fileDistrict.readlines()
    for line in content:
        line = line.strip()
        if len(line) >= 1 and line[0] != '%': # avoid comments and empty line
            MPdistrict.append(line)
