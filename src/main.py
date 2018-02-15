import numpy as np
import math
import RBF
import matplotlib.pyplot as plt

N = 0

def square(x):
    X = np.ones(len(x))
    for i in range(len(x)):
        if x[i] >= 0:
            X[i] = 1.0
        else:
            X[i] = -1.0
    return X

def generateDataset(noise = 0):
    N = math.ceil(2 * math.pi / 0.1)
    pts = np.linspace(0, 2*np.pi, num = N)
    train1 = np.sin(pts)
    train2 = square(train1)

    pts2 = np.linspace(0.05, 2*np.pi+0.05, num = N)
    test1 = np.sin(pts2)
    test2 = square(test1)

    return pts, pts2, train1, train2, test1, test2

Xtrain, Xtest, train1, train2, test1, test2 = generateDataset()

network1 = RBF.RBF_ite([0, 1/2 * np.pi, np.pi, 3/2 * np.pi, 2*np.pi], np.pi, 0.3, 2000)
eHisto = network1.fit(Xtrain, train1)
Y = network1.predict(Xtest)
print(eHisto[-1])

print(np.mean(abs(Y-test1)))

plt.plot(Xtest, Y)
plt.plot(Xtest, test1)
plt.show()

plt.plot(eHisto)
plt.show()


"""network2 = RBF.RBF_ite([0, 1, 10, 100, 1000, 10000, 100000], 1, 0.1, 700)
eHisto = network2.fit(Xtrain, train2)
Y = network2.predict(Xtest)

plt.plot(Xtest, Y)
plt.plot(Xtest, test2)
plt.show()

plt.plot(eHisto)
plt.show()"""
