# -*- coding: utf-8 -*-

import numpy as np
import copy

class SOM():

    def __init__(self, lr=0.001, nb_eboch=20, nb_hidden=100, neighbourhood_size=50):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.nb_hidden = nb_hidden
        self.W = []
        self.neighbourhood_size = neighbourhood_size

    def distance(self, V1, V2):
        return np.linalg.norm(V1-V2)

    def fit(self, X):
        self.W = np.random.normal(0, 1, (self.nb_hidden, X.shape[1]))
        print(self.W.shape)

        for step in range(self.nb_eboch):

            # For each input we want to find the closest unit
            for Xk in X:
                bestDistance = self.distance(Xk, self.W[0])
                bestUnit = 0

                for i, unit in enumerate(self.W[1:]):
                    dist = self.distance(Xk, unit)
                    if dist < bestDistance:
                        bestDistance = distance
                        bestUnit = i

                # We can update the weigths of the closest unit and its neighbourhood
                delta = 
