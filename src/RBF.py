import numpy as np


class RBF_ite:
    def __init__(self, means, variance, learningRate, epochNb):
        self.nodeNumber = len(means)
        self.means = means
        self.variances = [variance] * self.nodeNumber
        self.learningRate = learningRate
        self.weights = np.random.normal(0, 0.1, self.nodeNumber + 1).T
        self.epochNb = epochNb

    def fit(self, X, T):
        eHisto = []
        for epoch in range(self.epochNb):
            p = np.random.permutation(len(X))
            X, T = X[p], T[p]
            PHI = self.phi(X)
            f = np.dot(PHI, self.weights)

            for i in range(len(X)):
                f = np.dot(PHI, self.weights)
                e = (T[i] - f[i])
                deltaW = self.learningRate * e * PHI[i, :]
                self.weights += deltaW

            PHI = self.phi(X)
            f = np.dot(PHI, self.weights)
            e = sum(np.power(T - f, 2))
            eHisto.append(e)
        return eHisto


    def predict(self, X):
        PHI = self.phi(X)
        f = np.dot(PHI, self.weights)
        return f

    def phi(self, X):
        PHI = np.ones((len(X), self.nodeNumber + 1))
        for i in range(len(X)):
            for j in range(self.nodeNumber + 1):
                if j == 0:
                    PHI[i][j] = 1
                else:
                    PHI[i][j] = self.RBF_function(X[i], self.means[j-1], self.variances[j-1])
        return PHI

    def RBF_function(self, xi, mu, var):
        return np.exp( - (xi - mu) ** 2 / (2 * var))
