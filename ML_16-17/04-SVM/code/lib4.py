#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:57:08 2016

@author: markno1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import svm

    
pltt = plt

def plot_mesh(X_validation,clf,counter):
    h=0.01
    prediction = clf.predict(X_validation)
    x_min, x_max = X_validation[:, 0].min() - 1, X_validation[:, 0].max() + 1
    y_min, y_max = X_validation[:, 1].min() - 1, X_validation[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X_validation[:,0], X_validation[:,1] , c=prediction)
    plt.title('C=' +str(counter))
    plt.show()
    plt.close()
    
def plot_mesh_gamma(X_validation,clf,counter,counter2):
    h=0.01
    prediction = clf.predict(X_validation)
    x_min, x_max = X_validation[:, 0].min() - 1, X_validation[:, 0].max() + 1
    y_min, y_max = X_validation[:, 1].min() - 1, X_validation[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    #Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
    pltt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    pltt.scatter(X_validation[:,0], X_validation[:,1] , c=prediction)
    pltt.title('C=' +str(counter)+" Gamma= "+str(counter2))
   
    
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_density(X_validation,classifiers,lenght_C,lenght_Gamma,X,y):
    h=0.01
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        prediction = clf.predict(X_validation)
        x_min, x_max = X_validation[:, 0].min() - 1, X_validation[:, 0].max() + 1
        y_min, y_max = X_validation[:, 1].min() - 1, X_validation[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        plt.figure(figsize=(8, 6))
        
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # visualize decision function for these parameters
        plt.subplot(lenght_C, lenght_Gamma, k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),size='medium')
        # visualize parameter's effect on decision function
        cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
        plt.pcolormesh(xx, yy, -Z, cmap=cmap_light)
        plt.scatter(X_validation[:, 0], X_validation[:, 1], c=prediction)
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')
        
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(lenght_Gamma), gamma_range, rotation=45)
    plt.yticks(np.arange(lenght_C), C_range)
    plt.title('Validation accuracy')
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    