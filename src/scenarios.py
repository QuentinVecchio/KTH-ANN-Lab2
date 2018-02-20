# -*- coding: utf-8 -*-

import RBF
import graph
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neural_network import MLPRegressor

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

def posneg(X):
    Y = np.empty(len(X))
    for i,x in enumerate(X):
        if x >= 0:
            Y[i] = 1
        else:
            Y[i] = -1
    return Y

def scenario3_1():
    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    NB_HIDDEN=7# 8 , 12 ,18  #square = 5,10,25

    ##### Sinus Function #####
    YSinus = np.sin(2*X)
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = np.sin(2*XTest)
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=1000, nb_hidden=NB_HIDDEN, batch_size=1)
    history = rbfSinus.fit(X=X, T=YSinus, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))#np.sqrt(2*np.pi/NB_HIDDEN)
    YSinusRBF = rbfSinus.predict(XTest)
    print("Test error = "  + str(abs(np.sum((YSinusRBF - YSinusTest)))))
    graph.plotRBFInformations("Sinus function", XTest, YSinusTest, YSinusRBF, YSinus, history, [0, 2*np.pi, -1.5, 1.5])

    ##### Square Function #####
    YSquare = square(2*X)
    YSquare = YSquare.reshape((len(YSquare),1))
    YSquareTest = square(2*XTest)
    YSquareTest = YSquareTest.reshape((len(YSquareTest),1))
    rbfSquare = RBF.simpleRBF(lr=0.2, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    #history = rbfSquare.fit(X=X, T=YSquare, mu=[0, 2*np.pi, np.pi-0.01, np.pi+0.01, np.pi-0.05, np.pi+0.05] , sigma=np.sqrt(1))
    history = rbfSquare.fit(X=X, T=YSquare, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    YSquareRBF = rbfSquare.predict(XTest)
    print("Test error = "  + str(abs(np.sum((posneg(YSquareRBF) - YSquareTest)))))
    graph.plotRBFInformations("Square function", XTest, YSquareTest, YSquareRBF, YSquare, posneg(YSquareRBF), history, [0, 2*np.pi, -1.5, 1.5])

def scenario3_2():
    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    NB_HIDDEN= 8 #6, 11 , 26
    ##### Sinus Function #####

    YSinus = np.sin(2*X)
    # Add some noise
    error = np.random.normal(0, 0.1, len(YSinus))
    YSinus += error
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = np.sin(2*XTest)
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=150, nb_hidden=NB_HIDDEN, batch_size=1)
    history = rbfSinus.fit(X=X, T=YSinus, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=150, nb_hidden=NB_HIDDEN, batch_size=1)
    history2 = rbfSinus.fit(X=X, T=YSinus, mu=np.random.rand( 1,NB_HIDDEN)*2*np.pi , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    graph.plotRBFInformationsHistory("Error for different initialisation of Mu", history, history2, "uniform", "random", [0, 2*np.pi, -1.5, 1.5])

    #YSinusRBF = rbfSinus.predict(XTest)
    #print("Test error = "  + str(abs(np.sum((YSinusRBF - YSinusTest)))))
    #graph.plotRBFInformations("Sinus function with noise", XTest, YSinusTest, YSinusRBF, YSinus, history, [0, 2*np.pi, -1.5, 1.5])

    ##### Square Function #####
    # YSquare = square(2*X)
    # # Add some noise
    # error2 = np.random.normal(0, 0.1, len(YSquare))
    # YSquare += error2
    # YSquare = YSquare.reshape((len(YSquare),1))
    # YSquareTest = square(2*XTest)
    # YSquareTest = YSquareTest.reshape((len(YSquareTest),1))
    # rbfSquare = RBF.simpleRBF(lr=0.01, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    #history = rbfSquare.fit(X=X, T=YSquare, mu=[0, 2*np.pi, np.pi-0.01, np.pi+0.01, np.pi-0.05, np.pi+0.05] , sigma=np.sqrt(1))
    #history = rbfSquare.fit(X=X, T=YSquare, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    #YSquareRBF = rbfSquare.predict(XTest)
    #graph.plotRBFInformations("Square function with noise", XTest, YSquareTest, YSquareRBF, YSquare, history, [0, 2*np.pi, -1.5, 1.5])

def scenario3_2_1():

    N = math.ceil(2 * math.pi / 0.01)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    NB_HIDDEN= 12 #6, 11 , 26
    ##### Sinus Function #####

    YSinus = np.sin(2*X)

    error = np.random.normal(0, 0.1, len(YSinus))
    YSinus += error

    YSinusTest = np.sin(2*XTest)


    nn = MLPRegressor(
        hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
        random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    nn = MLPRegressor(solver='sgd', learning_rate_init=0.2, batch_size='auto', shuffle=False, hidden_layer_sizes=(8,4), max_iter=150)
    n = nn.fit(X.reshape(-1,1), YSinus.ravel())
    test_x = np.arange(0.0, 2 * np.pi, 0.05).reshape(-1, 1)
    test_y = nn.predict(test_x)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(X.reshape(-1,1), YSinus.ravel(), s=1, c='b', marker="s", label='real')
    ax1.scatter(test_x,test_y, s=10, c='r', marker="o", label='NN Prediction')
    plt.show()

    clf = MLPRegressor(solver='sgd', learning_rate_init=0.2, batch_size=100, shuffle=False, hidden_layer_sizes=(8,4), max_iter=1500)
    nn.fit(x,y)
    prediction1 = nn.predict(x)

    N = math.ceil(2 * math.pi / 0.1)
    X = np.linspace(0, 2*np.pi, num=N)
    XTest = np.linspace(0.05, 2*np.pi+0.05, num = N)
    YSinus = np.sin(2*X)
    YSinus += error
    YSinusTest = np.sin(2*XTest)
    YSinus = YSinus.reshape((len(YSinus),1))
    YSinusTest = YSinusTest.reshape((len(YSinusTest),1))

    rbfSinus = RBF.simpleRBF(lr=0.2, nb_eboch=150, nb_hidden=NB_HIDDEN, batch_size=1, verbose=False)
    history = rbfSinus.fit(X=X, T=YSinus, mu=np.linspace(0, 2*np.pi, num=NB_HIDDEN) , sigma=np.sqrt(2*np.pi/NB_HIDDEN))
    prediction2 = rbfSinus.predict(XTest)
    graph.plot3_2_1("",XTest, prediction1,prediction2,YSinusTest,[0, 2*np.pi, -1.5, 1.5])

def scenario3_3():
    N = math.ceil(2 * math.pi / 0.1)


    X1 = np.linspace(0.0, np.pi, num = 0/4)
    X2 = np.linspace(np.pi, 2* np.pi, num = N)
    X = []
    for x in X1:
        X.append(x)
    for x in X2:
        X.append(x)
    X = np.reshape(X,(N,1))

    NB_HIDDEN=10
    ##### Sinus Function #####

    YSinus = np.sin(2*X)
    YSinus = YSinus.reshape((len(YSinus),1))

    # Without Noise
    rbfSinus = RBF.simpleRBF(lr=0.4, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    history = rbfSinus.fit(X=X, T=YSinus , sigma=1)#np.sqrt(2*np.pi/NB_HIDDEN)


    bfSinus = RBF.simpleRBF(lr=0.4, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    rbfSinus.initialisationMu(X)
    history2 = rbfSinus.fit(X=X, T=YSinus , sigma=1)#np.sqrt(2*np.pi/NB_HIDDEN)

    # With Noise
    YSinus = np.sin(2*X)
    error = np.reshape(np.random.normal(0, 0.1, len(YSinus)), ( N,1))

    YSinus += error
    YSinus = YSinus.reshape((len(YSinus),1))

    rbfSinus = RBF.simpleRBF(lr=0.4, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    history3 = rbfSinus.fit(X=X, T=YSinus , sigma=1)#np.sqrt(2*np.pi/NB_HIDDEN)


    bfSinus = RBF.simpleRBF(lr=0.4, nb_eboch=500, nb_hidden=NB_HIDDEN, batch_size=1)
    rbfSinus.initialisationMu(X)
    history4 = rbfSinus.fit(X=X, T=YSinus , sigma=1)#np.sqrt(2*np.pi/NB_HIDDEN)
    print(history4)

    graph.plotRBFInformationsHistory("Comparaison", history, history2, history3,history4, [0, 2*np.pi, -1.5, 1.5])
    ##### Square Function #####
def ouverture1():
    fileTrain = open('ballist.dat', 'r')
    X_train = []
    Y_train = []
    for line in fileTrain:
        line = line.split("\t")
        x = line[0].split(" ")
        xToAdd = []
        xToAdd.append(float(x[0]))
        xToAdd.append(float(x[1]))
        X_train.append(xToAdd)
        y = line[1].split(" ")
        yToAdd = []
        yToAdd.append(float(y[0]))
        yToAdd.append(float(y[1][:-1]))
        Y_train.append(yToAdd)
    return np.reshape(X_train,(len(X_train),len(X_train[0]))), np.reshape(Y_train,(len(Y_train),len(Y_train[0])))

def ouverture2():
    fileTrain = open('balltest.dat', 'r')
    X_train = []
    Y_train = []
    for line in fileTrain:
        line = line.split("\t")
        x = line[0].split(" ")
        xToAdd = []
        xToAdd.append(float(x[0]))
        xToAdd.append(float(x[1]))
        X_train.append(xToAdd)
        y = line[1].split(" ")
        yToAdd = []
        yToAdd.append(float(y[0]))
        yToAdd.append(float(y[1][:-1]))
        Y_train.append(yToAdd)
    return np.reshape(X_train,(len(X_train),len(X_train[0]))), np.reshape(Y_train,(len(Y_train),len(Y_train[0])))

def scenario3_3_ballistic():
    NB_HIDDEN=10


    X, Y = ouverture1()
    RBFball = RBF.simpleRBFBall(lr=0.0001, nb_eboch=1000, nb_hidden=NB_HIDDEN, batch_size=1)
    RBFball.initialisationMu(X)
    history = RBFball.fit(X=X, T=Y , sigma=np.sqrt(2*np.pi/NB_HIDDEN))#np.sqrt(2*np.pi/NB_HIDDEN)
    Xtest, Ytest = ouverture2()
    print(RBFball.error(Xtest,Ytest))

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
