# -*- coding: utf-8 -*-

import numpy as np
import copy
import numpy as np

class simpleRBF():
#   Implements a RBF with 1 input and 1 output

    def __init__(self, lr=0.001, nb_eboch=20, batch_size=-1, nb_hidden=5):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.nb_hidden = nb_hidden
        self.W = []
        self.mu = []
        self.sigma = []

    def errorGivenPrediction(self, K, T):
        return sum(np.square(K-T))

    def error(self, X, T):
        return np.sum(np.square(self.predict(X)-T))

    def getMatricePhi(self, X):
        PHI = np.ones((len(X),self.nb_hidden+1))
        for i,Xk in enumerate(X):
            p = self.phi(Xk)
            p = np.insert(p,0,1)
            PHI[i] = p
        return PHI

    def phi(self,x):
        nominateur = -np.square(x-self.mu)
        denominateur = 2 * np.square(self.sigma)
        return np.exp(nominateur/denominateur)

    def fit(self, X=[], T=[], mu=[], sigma=0):
        self.W = np.random.normal(0, 1, self.nb_hidden+1)
        self.W = np.reshape(self.W, (self.nb_hidden+1,1))
        bestW = self.W[:]
        bestError = np.inf

        if len(mu) == 0:
            self.mu = np.reshape(np.random.uniform(low=-10, high=10, size=self.nb_hidden), (1, self.nb_hidden))
        else:
            self.mu = np.array(mu)
        self.sigma = np.array([sigma] * (self.nb_hidden))

        errorHistory = []
        PHI = self.getMatricePhi(X)

        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X))
            X, T, PHI = X[p], T[p], PHI[p]

            if step%(self.nb_eboch/10) == 0:
                print("Step : " + str(step))
                print("Current error : " + str(self.error(X,T)))

            for i,Xk in enumerate(X):
                phiXk = PHI[i]
                phiXk2 = phiXk.reshape((1,len(phiXk)))
                e = (T[i] - np.dot(phiXk, self.W))
                deltaW = self.lr * e * phiXk.T
                deltaW = np.reshape(deltaW, (self.nb_hidden+1,1))
                #print(deltaW.shape)
                self.W += deltaW

            if abs(self.error(X,T)) < bestError:
                bestError = abs(self.error(X,T))
                bestW = self.W[:]
            errorHistory.append(self.error(X,T))

        self.W = bestW
        print("Best error : " + str(bestError))
        return errorHistory

    def predict(self, X):
        PHI = self.getMatricePhi(X)
        return np.dot(PHI, self.W)
