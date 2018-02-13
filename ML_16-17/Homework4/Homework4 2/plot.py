#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:51:55 2016

@author: markno1
"""
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


cmap_light = (ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA']))

def plot(X_test,clf,title):
    res = 0.01
    plt.figure(figsize=(9,5))
    plt.title(title)
    plt.grid(True)
    
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),np.arange(y_min, y_max, res))
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    
 #   plt.contourf(xx, yy, Z, alpha=0.2, color=cmap_light)
    plt.pcolormesh(xx, yy, Z,alpha=0.2, cmap=cmap_light)
    
    plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=cmap_light)
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework3/plot/"+title+".png",dpi=100 )
    plt.legend(loc='upper left',shadow=True)
    plt.scatter(X_test[:,0],X_test[:,1],c=clf.predict(X_test),cmap = cmap_light)
    plt.show()