from __future__ import division
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
def plotRBFInformationsHistory(title, history, history2, history3, history4,axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.ylim([0, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])

    lines, = plt.plot(range(len(history)), history,'red')
    plt.plot(range(len(history)), history2,'green')
    plt.plot(range(len(history)), history3,'blue')
    plt.plot(range(len(history)), history4,'black')
    #plt.plot(range(len(history)), history5,'yellow')

    green_patch = mpatches.Patch(color='red', label="Sinus Approximation")
    red_patch = mpatches.Patch(color='green', label="Sinus Approximation with initialisation")
    blue_patch = mpatches.Patch(color='blue', label="Sinus Approximation with noise")
    black_patch = mpatches.Patch(color='black', label="Sinus Approximation with noise and initialisation ")
    #yellow_patch = mpatches.Patch(color='yellow', label=str(lr5))
    plt.legend(handles=[red_patch, green_patch, blue_patch, black_patch])#, yellow_patch])

    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve :  " + title)
    plt.show()

def plotRBFInformationsHistory3_2eta(title, history, history2, history3, history4, history5, history6,axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.ylim([0, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])

    lines, = plt.plot(range(len(history)), history,'red')
    plt.plot(range(len(history)), history2,'green')
    plt.plot(range(len(history)), history3,'blue')
    plt.plot(range(len(history)), history4,'black')
    plt.plot(range(len(history)), history5,'yellow')
    plt.plot(range(len(history)), history6,'purple')
    green_patch = mpatches.Patch(color='red', label="0.2")
    red_patch = mpatches.Patch(color='green', label="0.1")
    blue_patch = mpatches.Patch(color='blue', label="0.5")
    black_patch = mpatches.Patch(color='black', label="0.1 ")
    yellow_patch = mpatches.Patch(color='yellow', label="0.01")
    purple_patch = mpatches.Patch(color='purple', label="0.001")
    plt.legend(handles=[red_patch, green_patch, blue_patch, black_patch, yellow_patch, purple_patch])

    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve :  " + title)
    plt.show()

def plotError(title, eHistory):
    plt.ylim([-0.1, max(eHistory)*1.1])
    plt.xlim([-len(eHistory) * 0.1, len(eHistory) + len(eHistory) * 0.1])
    lines, = plt.plot(range(len(eHistory)), eHistory)
    plt.setp(lines, linewidth=2, color='r')
    plt.title(title)
    plt.show()

def plot3_2_1(title,X, history, history2, true ,axis):

    ig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.axis(axis)
    plt.title("RBF vs 2-layer on 150 epoch" + title)
    plt.plot(X, history, 'orange')
    plt.plot(X, history2, 'blue')
    plt.plot(X, true, 'green')
    #plt.plot(X, fT, 'black')
    red_patch = mpatches.Patch(color='blue', label='RBF')
    blue_patch = mpatches.Patch(color='orange', label='2-layer')
    green_patch = mpatches.Patch(color='green', label='true')

    #black_patch = mpatches.Patch(color='black', label='Approximation with transform')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.axis([0, 2*np.pi, -1.5, 1.5])
    plt.show()

def plotCl(title, X, T):
    X = X.T
    T = (T + 1) // 2

    x = 1
    y = 0
    bias = 2

    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    colors = ['red', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])

    #x = np.linspace(-5, 5, 50)

    ymin, ymax = plt.ylim()
    plt.title("plot" + title)
    plt.show()

def plotMuAnim(title, X, T, muHistory):
    fig, ax = plt.subplots(figsize=(12, 8))
    T = (T + 1) // 2
    X = X.T
    colors = ['red', 'blue']

    ax.scatter(X[:, 0], X[:, 1], c=[colors[i] for i in T])
    p1, = ax.plot(muHistory[0][0][0], muHistory[0][0][1], 'o', c="green", markersize=10)
    p2, = ax.plot(muHistory[0][1][0], muHistory[0][1][1], 'o', c="green", markersize=10)

    titleIteration = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def update_plot(i):
        titleIteration.set_text('Epoch Iteration ' + str(i))
        p1.set_data(muHistory[i][0][0], muHistory[i][0][1])
        p2.set_data(muHistory[i][1][0], muHistory[i][1][1])
        return p1, p2,

    ani = animation.FuncAnimation(fig, update_plot, frames=len(muHistory), interval=100)
    plt.title(title)
    plt.show()

def plotBallistic(X, mu):
    plt.subplot(2, 1, 1)
    # the histogram of the data
    plt.hist2d(X[:,0], X[:,1], bins=6, norm=LogNorm())
    plt.title("Representation of ballist data in 2D")
    plt.colorbar()
    plt.subplot(2, 1, 2)
    print(mu[:,0])
    plt.plot(mu[:,0],mu[:,1],'o')
    plt.title("Position of mu points")
    plt.axis([0,1,0,1])
    plt.show()

def plotTest(title, points):
    x,y = points.T
    plt.plot(x,y,'bo')
    plt.show()

def plotSOMAnimation(title, coordinates, WHistory):
    fig, ax = plt.subplots(figsize=(10, 6))

    x, y = coordinates.T
    ax.plot(x, y, 'o', c="green", markersize=10)

    x, y = WHistory[0].T
    mat, = ax.plot(x, y, 'o', c="blue", markersize=10)

    titleIteration = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def update_plot(i):
        titleIteration.set_text('Epoch Iteration ' + str(i)) #% len(WHistory)
        x, y = WHistory[i].T
        mat.set_data(x, y)
        return mat,

    ani = animation.FuncAnimation(fig, update_plot, frames=len(WHistory), interval=100)
    plt.title(title)
    plt.show()

def plotMap(title, coordinates, WHistory, result):
    fig = plt.figure(figsize=(8, 8))
    plt.title(title)
    x, y = coordinates.T
    #labels = ['{0}'.format(i) for i in range(len(coordinates))]
    plt.subplot(3, 1, 1)
    plt.plot(x, y, 'o', markersize=10)
    # for label, x, y in zip(labels, coordinates[:, 0], coordinates[:, 1]):
    #     plt.annotate(label, xy=(x, y), xytext=(-3, -3), textcoords='offset points')

    plt.subplot(3, 1, 2)
    xW, yW = WHistory[:-1].T
    plt.plot(xW, yW, 'o', markersize=10)

    plt.subplot(3, 1, 3)
    xResult, yResult = result.T
    plt.plot(xResult, yResult, 'o', markersize=10)

    for i in range(len(result)):
        n1 = result[i]
        n2 = result[(i+1)%len(result)]
        plt.plot([n1[0], n2[0]], [n1[1], n2[1]], 'black', zorder=0)

    plt.show()

def plot4_2(title, coordinates, WHistory, result):
    fig, axes = plt.subplots(2,2,figsize=(20, 10))
    plt.title(title)

    axes[0, 0].set_title("Cities")
    x, y = coordinates.T
    axes[0, 0].plot(x, y, 'o', c='g', markersize=10)

    axes[0, 1].set_title("TSP")
    xResult, yResult = result.T
    axes[0, 1].plot(x, y, 'o', c='g', markersize=10)
    axes[0, 1].plot(xResult, yResult, 'o', c='b', markersize=10)
    for i in range(len(result)):
        n1 = result[i]
        n2 = result[(i+1)%len(result)]
        axes[0, 1].plot([n1[0], n2[0]], [n1[1], n2[1]], 'black', zorder=0)

    axes[1, 0].set_title("Output at the end")
    xW, yW = WHistory[-1].T
    axes[1, 0].plot(xW, yW, 'o', c='orange', markersize=10)
    axes[1, 0].plot(x, y, 'o', c='g', markersize=10)

    axes[1, 1].set_title("Mapping animation")
    xW, yW = WHistory[0].T
    axes[1, 1].plot(x, y, 'o', c="green", markersize=10)
    mat, = axes[1, 1].plot(xW, yW, 'o', c="blue", markersize=10)

    titleIteration = axes[1, 1].text(0.05, 0.9, '', transform=axes[1, 1].transAxes)

    def update_plot(i):
        i=99
        titleIteration.set_text('Epoch Iteration ' + str(i)) #% len(WHistory)
        x, y = WHistory[i].T
        mat.set_data(x, y)
        return mat,

    ani = animation.FuncAnimation(fig, update_plot, frames=len(WHistory), interval=100)

    plt.show()

def plotMapWithColors(title, coordinates, colors):
    x, y = coordinates.T
    x = x + np.random.normal(0, 0.25, len(x))
    y = y + np.random.normal(0, 0.25, len(y))

    labels = ['{0}'.format(i) for i in range(len(coordinates))]
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(x,y,marker='.', s=100, c = colors)
    plt.title(title)
    plt.show()


def plotRBFInformationsHistory3_3(title, history, history2, history3, history4,axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)
    plt.ylim([0, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])

    lines, = plt.plot(range(len(history)), history,'red')
    plt.plot(range(len(history)), history2,'green')
    plt.plot(range(len(history)), history3,'blue')
    plt.plot(range(len(history)), history4,'black')
    #plt.plot(range(len(history)), history5,'yellow')

    green_patch = mpatches.Patch(color='red', label="Sinus Approximation")
    red_patch = mpatches.Patch(color='green', label="Sinus Approximation with initialisation")
    blue_patch = mpatches.Patch(color='blue', label="Sinus Approximation with noise")
    black_patch = mpatches.Patch(color='black', label="Sinus Approximation with noise and initialisation ")
    #yellow_patch = mpatches.Patch(color='yellow', label=str(lr5))
    plt.legend(handles=[red_patch, green_patch, blue_patch, black_patch])#, yellow_patch])

    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve :  " + title)
    plt.show()

def plotRBFInformations3_2eta(title, X, f, approx, approx2, approx3, approx4, approx5, approx6, axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(1, 1, 1)

    plt.axis(axis)
    plt.title("RBF for " + title)
    plt.plot(X, f, 'green')
    plt.plot(X, approx, 'red')
    plt.plot(X, approx2, 'orange')
    plt.plot(X, approx3, 'blue')
    plt.plot(X, approx4, 'black')
    plt.plot(X, approx5, 'yellow')
    plt.plot(X, approx6, 'purple')
    
    green_patch = mpatches.Patch(color='green', label='Test')
    red_patch = mpatches.Patch(color='red', label="0.5")
    orange_patch = mpatches.Patch(color='orange', label="0.2")
    blue_patch = mpatches.Patch(color='blue', label="0.1")
    black_patch = mpatches.Patch(color='black', label="0.05")
    yellow_patch = mpatches.Patch(color='yellow', label="0.01")
    purple_patch = mpatches.Patch(color='purple', label="0.001")
    
    plt.legend(handles=[green_patch, red_patch, orange_patch,blue_patch, black_patch, yellow_patch, purple_patch])#, blue_patch])
    plt.axis([0, 2*np.pi, -1.5, 1.5])
    plt.show()
   


def plotRBFInformations3_2(title,X, test, approx, true, history, axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)

    plt.axis(axis)
    plt.title("RBF for " + title)
    plt.plot(X, test, 'green')
    plt.plot(X, approx, 'red')
    plt.plot(X, true, 'orange')
    green_patch = mpatches.Patch(color='green', label='Test')
    red_patch = mpatches.Patch(color='red', label="Approximation")
    orange_patch = mpatches.Patch(color='orange', label="True")
    plt.legend(handles=[green_patch, red_patch, orange_patch])
    plt.axis([0, 2*np.pi, -1.5, 1.5])
    plt.subplot(2, 1, 2)
    plt.ylim([-0.1, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])
    lines, = plt.plot(range(len(history)), history)
    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve  " + title)
    plt.show()
def plotRBFInformations3_1(title,X, test, approx, true, posneg, history, axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)

    plt.axis(axis)
    plt.title("RBF for " + title)
    plt.plot(X, test, 'green')
    plt.plot(X, approx, 'red')
    plt.plot(X, true, 'orange')
    plt.plot(X, posneg, 'black')
    green_patch = mpatches.Patch(color='green', label='Test')
    red_patch = mpatches.Patch(color='red', label="Approximation")
    orange_patch = mpatches.Patch(color='orange', label="True")
    black_patch = mpatches.Patch(color='black', label="Transform")
    plt.legend(handles=[green_patch, red_patch, orange_patch, black_patch])
    plt.axis([0, 2*np.pi, -1.5, 1.5])
    plt.subplot(2, 1, 2)
    plt.ylim([-0.1, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])
    lines, = plt.plot(range(len(history)), history)
    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve  " + title)
    plt.show()