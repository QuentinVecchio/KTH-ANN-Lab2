# -*- coding: utf-8 -*-

import numpy as np
import copy

class SOM():

    def __init__(self, lr=0.001, nb_eboch=20, output=[1,100], neighbourhood_size=50):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.output = output
        self.W = []
        self.neighbourhood_size = neighbourhood_size

    def distance(self, V1, V2):
        return np.linalg.norm(V1-V2)

    def distanceManhattan(self, V1, V2):
        return np.linalg.norm(V1-V2, ord=1)

    def neighbourhoodByIndex(self, winner):
        neighbourhood = [winner]

        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            if (winner[1]-i) < 0:
                break
            neighbourhood.append([0,(winner[1]-i)])

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            if (i+winner[1]) >= len(self.W):
                break
            neighbourhood.append([0,i+winner[1]])

        return neighbourhood

    def neighbourhoodByIndexCircular(self, winner):
        neighbourhood = [winner]

        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            index = winner[1]-i
            if index < 0:
                i = len(self.W) - 1 -i
            neighbourhood.append([0,index])

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            neighbourhood.append([0,(i+winner[1])%len(self.W)])

        return neighbourhood

    def neighbourhoodByManhattan(self, winner):
        neighbourhood = []

        # TODO : improve this code
        for x in range(self.W.shape[0]):
            for y in range(self.W.shape[1]):
                if self.distanceManhattan(winner, self.W[x][y]) <= self.neighbourhood_size:
                    neighbourhood.append([x,y])

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

    def predictWinnerForInput(self, input):
        bestDistance = self.distance(input, self.W[0][0])
        winner = (0,0)

        for x in range(self.W.shape[0]):
            for y in range(self.W.shape[1]):
                dist = self.distance(input, self.W[x][y])
                if dist < bestDistance:
                    bestDistance = dist
                    winner = (x,y)

        return winner

    def fit(self, X, method):
        firts_neighbourhood_size = self.neighbourhood_size
        self.W = np.random.uniform(0, 1, (self.output[0], self.output[1], X.shape[1]))
        print(self.W.shape) # 10,15,31

        for step in range(self.nb_eboch):
            if step%(self.nb_eboch/10) == 0:
                print("Epoch " + str(step) + " ...")

            # For each input we want to find the closest unit
            for Xk in X:
                winner = self.predictWinnerForInput(Xk)

                neighbourhood = []
                if method == "index":
                    neighbourhood = self.neighbourhoodByIndex(winner)
                if method == "index-circular":
                    neighbourhood = self.neighbourhoodByIndexCircular(winner)
                if method == "manhattan":
                    neighbourhood = self.neighbourhoodByManhattan(winner)
                # elif method == "distance":
                #     neighbourhood = self.neighbourhoodByDistance(winner)
                # elif method == "size":
                #     neighbourhood = self.neighbourhoodBySize(winner)
                #     neighbourhood = [i[0] for i in neighbourhood]

                # Update the weight of neighbourhood
                for i in neighbourhood:
                    x = i[0]
                    y = i[1]
                    deltaW = self.lr * (Xk - self.W[x][y])
                    self.W[x][y] += deltaW

            # We update the neighbourhood_size to be close of 1
            self.neighbourhood_size = self.neighbourhood_size - (firts_neighbourhood_size) / self.nb_eboch

    def predict(self, X):
        result = []
        for index,Xk in enumerate(X):
            winner = self.predictWinnerForInput(Xk)
            result.append((index,winner))

        return result
