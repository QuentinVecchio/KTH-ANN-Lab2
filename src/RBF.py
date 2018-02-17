# -*- coding: utf-8 -*-

import numpy as np
import copy
import numpy as np

class simpleRBF():
#   Implements a RBF with 1 input and 1 output

    def __init__(self, lr=0.001, nb_eboch=20, batch_size=1, nb_hidden=5):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.nb_hidden = nb_hidden
        self.W = []
        self.mu = np.reshape(np.array(np.linspace(0, 2*np.pi, num=self.nb_hidden)) , (1,self.nb_hidden))
        self.sigma = []
    def initialisationMu(self,X, nb_ite=100):
        print(self.mu.shape)
        p = np.random.randint(len(X),size=nb_ite)
        for index in p:
            #print(index)
            current = X[index]
            best_index = 0;
            best_value = 100000000.0;
            for index_mu in range(len(self.mu[0])):
                node_mu = self.mu[0][index_mu]
                value = np.linalg.norm([current - node_mu], ord=2)
                if(value < best_value):

                    best_value = value
                    best_index = index_mu

            self.mu[0][best_index] = self.mu[0][best_index] + self.lr * (current - self.mu[0][best_index])
        print(self.mu)
    def errorGivenPrediction(self, K, T):
        return sum(np.square(K-T))

    def error(self, X, T):
        #print(str(self.predict(X).shape) + " " + str(T.shape))
        return np.sum((self.predict(X)-T))# / len(X)*1.0

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

        
        if(len(mu) >0): 
            self.mu = np.array(mu)
        self.sigma = np.array([sigma] * (self.nb_hidden))
        print(self.mu)
        errorHistory = []
        PHI = self.getMatricePhi(X)

        for step in range(self.nb_eboch):
            p = np.random.permutation(len(X))
            X, T, PHI = X[p], T[p], PHI[p]

            if step%(self.nb_eboch/10) == 0:
                print("Step : " + str(step))
                print("Current error : " + str(abs(self.error(X,T))))

            batchIndex_list =[]
            if(self.batch_size == -1):
                batchIndex_list.append([0,len(X)])
            else:
                for i in range(int((len(X) * 1.0) / self.batch_size)):
                    batchIndex_list.append([i * self.batch_size, (i + 1) * self.batch_size])

            # for i,Xk in enumerate(X):
            #     phiXk = PHI[i]
            #     phiXk2 = phiXk.reshape((1,len(phiXk)))
            #     e = (T[i] - np.dot(phiXk, self.W))
            #     deltaW = self.lr * e * phiXk.T
            #     deltaW = np.reshape(deltaW, (self.nb_hidden+1,1))
            #     self.W += deltaW

            for batchIndex in batchIndex_list:
                start, end = batchIndex
                batchX = X[start:end] #(end-start + 1) x 1
                phiXk = PHI[start:end]# (end-start + 1) x n
                #print(phiXk.shape)
                e = (T[start:end] - np.dot(phiXk, self.W))
                #print("error shape : " + str(e.shape))
                deltaW = self.lr  * np.dot(phiXk.T,e)

                deltaW = np.reshape(deltaW, (self.nb_hidden + 1,1))
                #print(deltaW.shape)
                self.W += deltaW
            errorAux = abs(self.error(X,T))
            if abs(errorAux) < bestError:
                bestError = abs(errorAux)
                bestW = self.W[:]
            errorHistory.append(errorAux)

        self.W = bestW
        print("Best error : " + str(bestError))
        return errorHistory

    def predict(self, X):
        PHI = self.getMatricePhi(X)
        return np.dot(PHI, self.W)
