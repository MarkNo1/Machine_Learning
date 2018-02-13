#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:32:07 2016

@author: markno1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#colors = plt.cm.cool
colors = (ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA', '#FCFFAA']))


class plotting_grid():
    my_plt = plt
    number_of_plot = 0
    grid_r = 0
    grid_c = 0

    def __init__(self, fig_r, fig_c, grid_r, grid_c):
        self.number_of_plot = 0
        self.my_plt = plt
        self.grid_r = grid_r
        self.grid_c = grid_c
        self.my_plt.figure(figsize=(fig_c, fig_r))

    def plot_2D_decision_regions_Kmean(self, X, kmeans, res, title, number_plot):

        self.my_plt.subplot(self.grid_r, self.grid_c, number_plot + 1)
        self.my_plt.title(title)
        self.my_plt.grid(True)
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        Z = kmeans.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        self.my_plt.contourf(xx, yy, Z, alpha=0.2, cmap=colors)
        self.my_plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), cmap=colors)
        centroids = kmeans.cluster_centers_
        self.my_plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=169, linewidths=3, color='r', zorder=10)

    def save(self, title):
        self.my_plt.savefig("../plot/" + title + ".png", dpi=100)

    def show(self):
        self.my_plt.show()
        self.my_plt.close()

    def plot_2D_decision_regions_GMM(self, X, gmm, res, title, number_plot):
        self.my_plt.subplot(self.grid_r, self.grid_c, number_plot + 1)
        self.my_plt.title(title)
        self.my_plt.grid(True)
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                             np.arange(y_min, y_max, res))

        Z = gmm.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        self.my_plt.contourf(xx, yy, Z, alpha=0.2, cmap=colors)
        self.my_plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap=colors)

        centroids = gmm.means_
        self.my_plt.scatter(centroids[:, 0], centroids[:, 1],
                            marker='x', s=169, linewidths=3, color='r', zorder=10)

    def plot_2D(self, X, label, title):
        plt.figure(figsize=(9, 5))
        plt.grid(True)
        plt.title(title)
        plt.scatter(X[:, 0], X[:, 1], c=label, cmap=colors)
        plt.savefig("../plot/" + title + ".png", dpi=100)
        plt.legend(loc='upper left', shadow=True)
        plt.show()
        plt.close()


def plot_info(info, title, init_x):

    plt.figure(figsize=(9, 5))
    inf = np.asarray(info)
    x = np.linspace(init_x, len(info) + 1, len(info))
    plt.plot(x, inf[:, 0], c="green", label="NMI")
    plt.plot(x, inf[:, 1], c="red", label="Homogenity")
    plt.plot(x, inf[:, 2], c="blue", label="Purity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.subplots_adjust(right=0.75)
    plt.savefig("../plot/" + title + ".png", dpi=100)
    plt.show()
