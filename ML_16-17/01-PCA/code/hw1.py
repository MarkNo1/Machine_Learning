#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 15:45:47 2016

@author: markno1
"""
from lib1 import *
from classificator import *
from sklearn.decomposition import PCA

# load images data
X = convertImg_matrix(load_class())

# Standardize
X = standardize(X)

# Principal Componet Alalisys - 1
pca = PCA(n_components=2)
X_t = pca.fit_transform(X)
plot(X_t, 0, 1)  # PLOT

# Principal Componet Alalisys - 2
pca = PCA(n_components=5)
X_t1 = pca.fit_transform(X)
plot(X_t1, 3, 4)  # PLOT

# Principal Componet Alalisys - 3
pca = PCA(n_components=12)
X_t2 = pca.fit_transform(X)
plot(X_t2, 10, 11)  # PLOT


# Classificator
cl1 = classificator(X_t, y(), "1 and 2 Component")
cl1.train_test()
cl1.gaussianNB()
cl1.plot()

X_t1 = X_t1[:, [3, 4]]
cl2 = classificator(X_t1, y(), "3 and 4 Component")
cl2.train_test()
cl2.gaussianNB()
cl2.plot()

X_t2 = X_t2[:, [10, 11]]
cl3 = classificator(X_t2, y(), "10 and 11 Component")
cl3.train_test()
cl3.gaussianNB()
cl3.plot()
