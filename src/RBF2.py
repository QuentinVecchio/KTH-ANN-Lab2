# -*- coding: utf-8 -*-

import numpy as np
import copy
import numpy as np

class simpleRBFBall2D():
#   Implements a RBF with 1 input and 1 output

    def __init__(self, lr=0.001, nb_eboch=20, batch_size=1, nb_hidden=5):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.nb_hidden = nb_hidden
        self.W = []
        self.mu = np.reshape(np.array(np.linspace(0, 1, num=2 * self.nb_hidden)) , (2, self.nb_hidden))
        self.sigma = np.sqrt(2.0 * np.pi / self.nb_hidden)
    def initialisationMu(self,X, nb_ite=10000):
        self.mu = np.reshape(np.random.rand( 1, 2 * self.nb_hidden), (2, self.nb_hidden))
        p = np.random.randint(len(X), size=nb_ite)
        frequency = np.zeros(self.nb_hidden)
        count = 1.0
        for index in p:
            current = X[index] # 2 number
            best_index = 0;
            best_value = np.finfo(np.float64).max

            for index_mu in range(len(self.mu[0])):
                node_mu = self.mu[:, index_mu]
                value = (frequency[index_mu] / count) ** 2 * np.linalg.norm([current - node_mu], ord=2)
                if(value < best_value):

                    best_value = value
                    best_index = index_mu
            count += 1
            frequency[best_index] += 1
            self.mu[:,best_index] = self.mu[:,best_index] + self.lr * (current - self.mu[:,best_index])
        return self.mu

    def errorGivenPrediction(self, e):
        error = 0.0
        for i in range(len(e)):
            error +=np.sqrt(e[i][0] ** 2 + e[i][1] ** 2)
        return error

    def error(self, X, T, flag=False):
        PHI = self.getMatricePhi(X)
        e = (T - np.dot(PHI,self.W.T))
        return self.errorGivenPrediction(e)
    def getMatricePhi(self, X):
        PHI = np.ones((len(X),self.nb_hidden+1))
        self.mu = np.reshape(self.mu,(2, self.nb_hidden))
        for i,Xk in enumerate(X):
            p = self.phi(Xk)
            p = np.insert(p,0,1)
            PHI[i] = p

        return PHI

    def phi(self,x):
        a = np.square(x[0] - np.reshape(self.mu[0],(1, self.nb_hidden)))
        b = np.square(x[1] - np.reshape(self.mu[1],(1, self.nb_hidden)))
        nominateur = np.sqrt(a + b)
        denominateur = 2 * np.square(self.sigma)
        return np.exp(nominateur/denominateur)
    def fit(self, X=[], T=[], mu=[], sigma=0):
        PHI = self.getMatricePhi(X)
        self.W = np.array(np.reshape(np.random.rand( 1, 2 * (self.nb_hidden + 1)), (2, (self.nb_hidden + 1))))
        bestError = 10000000.0
        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X))
            X, T, PHI = X[p], T[p], PHI[p]
            batchIndex_list =[]
            if step%(self.nb_eboch/10) == 0 :
                print("Step : " + str(step))
                print("Current error : " + str(self.error(X,T)))

            if(self.batch_size == -1):
                batchIndex_list.append([0,len(X)])
            else:
                for i in range(int((len(X) * 1.0) / self.batch_size)):
                    batchIndex_list.append([i * self.batch_size, (i + 1) * self.batch_size])

            for batchIndex in batchIndex_list:
                
                start, end = batchIndex
                batchX = X[start:end] #(end-start + 1) x 1
                phi = PHI[start:end]# (end-start + 1) x n
                e = (T[start:end] - np.dot(phi,self.W.T))
                error = self.errorGivenPrediction(e)
                deltaW = self.lr * np.dot(e.T,phi)
                self.W += deltaW
            errorAux = abs(self.error(X,T))
            if abs(errorAux) < bestError:
                bestError = abs(errorAux)
                bestW = np.copy(self.W)
        self.W = bestW
        return 

        
    def predict(self, X):
        PHI = self.getMatricePhi(X)
        return np.dot(PHI,self.W.T)

