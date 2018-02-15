from __future__ import division
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

def plotRBFInformations(title, X, f, fK, history, axis):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.axis(axis)
    plt.title("RBF for " + title)
    plt.plot(X, f, 'green')
    plt.plot(X, fK, 'red')
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
