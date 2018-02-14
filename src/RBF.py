# -*- coding: utf-8 -*-

import numpy as np
import copy

class simpleRBF():
#   Implements a RBF with 1 input and 1 output

    def __init__(self, lr=0.001, nb_eboch=20, batch_size=-1, nb_hidden=1):
        self.batch_size = batch_size
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.nb_hidden = nb_hidden
        self.W = []
        self.mu = []
        self.delta = []

    def errorGivenPrediction(self, K, T):
        return sum(pow(K-T,2))

    def error(self, X, T):
        return sum(pow(predict(X)-T,2))

    def phi(self,X):
        return exp((-pow(X-self.mu,2))/(2*pow(self.delta,2)))

    def fit(self, X, T):
        print("Not implemented yet")

    def predict(self, X):
        return sum(self.W * phi(X))
