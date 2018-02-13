#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:15:41 2016

@author: markno1
"""
import matplotlib.pyplot as plt
import numpy as np

colors = plt.cm.cool


def plot(X, X_test, X_train, clf, title):
    res = 0.01
    plt.figure(figsize=(9, 5))
    plt.title(title)
    plt.grid(True)
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, color=colors)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=colors)
    plt.savefig("../plot/" + title + ".png", dpi=100)
    plt.legend(loc='upper left', shadow=True)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=colors)
    plt.show()


def my_func(d):
    a = 0.1
    return np.exp(-a * (d**2))


def my_func1(d):
    a = 10
    return np.exp(-a * (d**2))


def my_func2(d):
    a = 100
    return np.exp(-a * (d**2))


def my_func3(d):
    a = 1000
    return np.exp(-a * (d**2))
