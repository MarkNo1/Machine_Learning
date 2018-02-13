#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:32:07 2016

@author: markno1
"""
import numpy as np
import matplotlib.pyplot as plt

colors = plt.cm.cool


def pur(confusion):
    r = len(confusion)
    pur = 0

    for i in range(r):
        pur += confusion[i].max()

    return pur / confusion.sum()


def plot_info(info):
    inf = np.asarray(info)
    x = np.linspace(2, len(info), len(info))
    plt.plot(x, inf[:, 0], c="green", label="NMI")
    plt.plot(x, inf[:, 1], c="red", label="Homogenity")
    plt.plot(x, inf[:, 2], c="blue", label="Purity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


# Input
#      X      -  Data
#      clf    -  classificator
#      res    -  Resolution of the boundary
#      title  -  Title

def plot_2D_decision_regions(X, clf, res, title):
    plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.grid(True)
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.01, color=colors)
    plt.scatter(X[:, 0], X[:, 1], c=clf.predict(X), cmap=colors)

    plt.savefig("../plot/" + title + ".png", dpi=100)
    plt.legend(loc='upper left', shadow=True)
    plt.show()
    plt.close()
