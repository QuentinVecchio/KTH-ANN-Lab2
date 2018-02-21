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
            if (i+winner[1]) >= len(self.W[0]):
                break
            neighbourhood.append([0,i+winner[1]])

        return neighbourhood

    def neighbourhoodByIndexCircular(self, winner):
        neighbourhood = [winner]

        #1 2 3 4 5
        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            index = winner[1]-i
            if index < 0:
                index = len(self.W[0]) - i
            neighbourhood.append([0,index])

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            neighbourhood.append([0,(i+winner[1])%len(self.W[0])])

        return neighbourhood

    def neighbourhoodByManhattan(self, winner):
        neighbourhood = []

        mat = np.ones((self.W.shape[0], self.W.shape[1]))

        # TODO : improve this code
        for x in range(self.W.shape[0]):
            for y in range(self.W.shape[1]):
                mat[x][y] = self.distanceManhattan(self.W[winner[0], winner[1]], self.W[x][y])

        m = np.max(mat)

        for i in range(int(self.neighbourhood_size)*2+1):
            t =  np.argmin(mat, axis=1)
            neighbourhood.append(t)
            mat[t[0]][t[1]] = m


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
        winner = [0,0]

        for x in range(self.W.shape[0]):
            for y in range(self.W.shape[1]):
                dist = self.distance(input, self.W[x][y])
                if dist < bestDistance:
                    bestDistance = dist
                    winner = [x,y]

        return winner

    def fit(self, X, method):
        self.output = np.array(self.output)
        firts_neighbourhood_size = self.neighbourhood_size
        self.W = np.random.uniform(0.1, 0.9, (self.output[0], self.output[1], X.shape[1]))
        print(self.W.shape) # 10,15,31
        WHistory = []
        frequency = np.zeros((self.W.shape[0], self.W.shape[1]))
        count = 1.0
        debug = False
        for step in range(self.nb_eboch):
            if step%(self.nb_eboch/10) == 0:
                print("Epoch " + str(step) + " ...")

            # For each input we want to find the closest unit
            for Xk in X:
                bestDistance = self.distance(Xk, self.W[0][0])
                winner = [0,0]

                for x in range(self.W.shape[0]):
                    for y in range(self.W.shape[1]):
                        # if self.neighbourhood_size > 0.2:
                        #     dist = ((frequency[x][y] / count)**2) * self.distance(Xk, self.W[x][y])
                        # else:
                        #     if not debug:
                        #         print("ok")
                        #         self.lr *= 2
                        #         debug = True
                        dist = self.distance(Xk, self.W[x][y])
                        if dist < bestDistance:
                            bestDistance = dist
                            winner = [x,y]

                count += 1
                frequency[winner[0]][winner[1]] += 1
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

            WHistory.append(copy.copy(self.W))

            # We update the neighbourhood_size to be close of 1
            self.neighbourhood_size = self.neighbourhood_size - (firts_neighbourhood_size) / self.nb_eboch
        return WHistory

    def predict(self, X):
        result = []
        coordinates = []
        for index,Xk in enumerate(X):
            winner = self.predictWinnerForInput(Xk)
            result.append(winner)
            coordinates.append((winner, self.W[winner[0]][winner[1]]))

        return result, coordinates

class SOM_TSP():

    def __init__(self, lr=0.001, nb_eboch=20, output=[1,100], neighbourhood_size=50):
        self.lr = lr
        self.nb_eboch = nb_eboch
        self.output = output
        self.W = []
        self.neighbourhood_size = neighbourhood_size

    def distance(self, V1, V2):
        return np.linalg.norm(V1-V2)

    def neighbourhoodByIndexCircular(self, winner):
        neighbourhood = [winner]

        #1 2 3 4 5
        # Add the left neighbourhood
        for i in range(int(self.neighbourhood_size)):
            index = winner[1]-i
            if index < 0:
                index = len(self.W[0]) - i
            neighbourhood.append([0,index])

        # Add the right neighbourhood
        for i in range(int(self.neighbourhood_size)):
            neighbourhood.append([0,(i+winner[1])%len(self.W[0])])

        return neighbourhood

    def predictWinnerForInput(self, input):
        bestDistance = self.distance(input, self.W[0][0])
        winner = [0,0]

        for x in range(self.W.shape[0]):
            for y in range(self.W.shape[1]):
                dist = self.distance(input, self.W[x][y])
                if dist < bestDistance:
                    bestDistance = dist
                    winner = [x,y]

        return winner

    def fit(self, X, method):
        self.output = np.array(self.output)
        firts_neighbourhood_size = self.neighbourhood_size
        self.W = np.empty((self.output[0], self.output[1], X.shape[1]))
        ## Warning HARD code
        N = self.output[1]
        R = 0.5
        x, y = X.T
        centroid = (sum(x) / len(X), sum(y) / len(X))
        print(centroid)
        for p in range(N):
            t = p*(2 * np.pi / N)
            self.W[0][p] = [R * np.cos(t) + centroid[0], R * np.sin(t) + centroid[1]]
        print(self.W.shape) # 10,15,31
        WHistory = []
        frequency = np.zeros((self.W.shape[0], self.W.shape[1]))
        count = 1.0
        debug = False
        for step in range(self.nb_eboch):
            if step%(self.nb_eboch/10) == 0:
                print("Epoch " + str(step) + " ...")

            # For each input we want to find the closest unit
            for Xk in X:
                bestDistance = self.distance(Xk, self.W[0][0])
                winner = [0,0]

                for x in range(self.W.shape[0]):
                    for y in range(self.W.shape[1]):
                        if self.neighbourhood_size > 0.2:
                            dist = ((frequency[x][y] / count)**2) * self.distance(Xk, self.W[x][y])
                        else:
                            if not debug:
                                print("CL finished")
                                self.lr *= 2
                                debug = True
                            dist = self.distance(Xk, self.W[x][y])
                        if dist < bestDistance:
                            bestDistance = dist
                            winner = [x,y]

                count += 1
                frequency[winner[0]][winner[1]] += 1
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

            WHistory.append(copy.copy(self.W))

            # We update the neighbourhood_size to be close of 1
            self.neighbourhood_size = self.neighbourhood_size - (firts_neighbourhood_size) / self.nb_eboch
        return WHistory

    def predict(self, X):
        result = []
        coordinates = []
        for index,Xk in enumerate(X):
            winner = self.predictWinnerForInput(Xk)
            result.append(winner)
            coordinates.append((winner, self.W[winner[0]][winner[1]]))

        return result, coordinates
