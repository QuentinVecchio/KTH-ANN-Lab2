from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

def plotRBFInformations(title, X, f, fK, fLearn, history, axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.axis(axis)
    plt.title("RBF for " + title)
    plt.plot(X, f, 'green')
    plt.plot(X, fK, 'red')
    plt.plot(X, fLearn, 'blue')
    red_patch = mpatches.Patch(color='red', label='f')
    green_patch = mpatches.Patch(color='green', label='fK')
    blue_patch = mpatches.Patch(color='blue', label='t')
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.axis([0, 2*np.pi, -1.5, 1.5])

    plt.subplot(2, 1, 2)
    plt.ylim([-0.1, max(history)*1.1])
    plt.xlim([-len(history) * 0.1, len(history) + len(history) * 0.1])
    lines, = plt.plot(range(len(history)), history)
    plt.setp(lines, linewidth=2, color='r')
    plt.title("Learning Curve  " + title)
    plt.show()

def plotError(title, eHistory):
    plt.ylim([-0.1, max(eHistory)*1.1])
    plt.xlim([-len(eHistory) * 0.1, len(eHistory) + len(eHistory) * 0.1])
    lines, = plt.plot(range(len(eHistory)), eHistory)
    plt.setp(lines, linewidth=2, color='r')
    plt.title(title)
    plt.show()

def plotMap(title, coordinates):
    x, y = coordinates.T
    labels = ['{0}'.format(i) for i in range(len(coordinates))]
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(x,y,marker='o', s=300)
    for label, x, y in zip(labels, coordinates[:, 0], coordinates[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-3, -3), textcoords='offset points')

    plt.title(title)
    plt.show()
