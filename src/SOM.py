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

    def neighbourhoodByIndex(self, winner):
        neighbourhood = [winner]

        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            if (winner-i) < 0:
                break
            neighbourhood.append((winner-i))

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            if (i+winner) >= len(self.W):
                break
            neighbourhood.append(i+winner)

        return neighbourhood

    def neighbourhoodByIndexCircular(self, winner):
        neighbourhood = [winner]

        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            index = winner-i
            if index < 0:
                i = len(self.W) - 1 -i
            neighbourhood.append(index)

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            neighbourhood.append((i+winner)%len(self.W))

        return neighbourhood

    def neighbourhoodByDistance(self, winner):
        neighbourhood = [winner]
        for i, unit in enumerate(self.W):
            if i != winner:
                dist = self.distance(self.W[winner], unit)
                if dist <= self.neighbourhood_size:
                    neighbourhood.append(i)
        return neighbourhood

    def neighbourhoodBySize(self, winner):
        neighbourhood = [(winner, 0)]
        for i, unit in enumerate(self.W):
            if i != winner:
                dist = self.distance(self.W[winner], unit)
                neighbourhood.append((i,dist))

        neighbourhood.sort(key=lambda tup: tup[1])
        neighbourhood = neighbourhood[:int(self.neighbourhood_size)]
        return neighbourhood

    def fit(self, X, method):
        firts_neighbourhood_size = self.neighbourhood_size
        self.W = np.random.uniform(0, 1, (self.nb_hidden, X.shape[1]))
        print(self.W.shape) # 100,84

        for step in range(self.nb_eboch):
            if step%(self.nb_eboch/10) == 0:
                print("Epoch " + str(step) + " ...")

            # For each input we want to find the closest unit
            for Xk in X:
                bestDistance = self.distance(Xk, self.W[0])
                winner = 0

                for i, unit in enumerate(self.W[1:]):
                    dist = self.distance(Xk, unit)
                    if dist < bestDistance:
                        bestDistance = dist
                        winner = i

                neighbourhood = []
                if method == "index":
                    neighbourhood = self.neighbourhoodByIndex(winner)
                if method == "index-circular":
                    neighbourhood = self.neighbourhoodByIndexCircular(winner)
                elif method == "distance":
                    neighbourhood = self.neighbourhoodByDistance(winner)
                elif method == "size":
                    neighbourhood = self.neighbourhoodBySize(winner)
                    neighbourhood = [i[0] for i in neighbourhood]

                # Update the weight of neighbourhood
                for i in neighbourhood:
                    deltaW = self.lr * (Xk - self.W[i])
                    self.W[i] += deltaW

            # We update the neighbourhood_size to be close of 1
            self.neighbourhood_size = self.neighbourhood_size - (firts_neighbourhood_size) / self.nb_eboch

    def predict(self, X):
        result = []
        for index,Xk in enumerate(X):
            bestDistance = self.distance(Xk, self.W[0])
            winner = 0

            for i, unit in enumerate(self.W[1:]):
                dist = self.distance(Xk, unit)
                if dist < bestDistance:
                    bestDistance = dist
                    winner = i
            result.append((index,winner))

        return result
