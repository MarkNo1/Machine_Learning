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

class plotting_grid():
    my_plt = plt
    number_of_plot = 0
    grid_r = 0
    grid_c =0

    def __init__(self,fig_r,fig_c,grid_r,grid_c):
        self.number_of_plot = 0
        self.my_plt = plt
        self.grid_r = grid_r
        self.grid_c = grid_c
        self.my_plt.figure(figsize=(fig_c,fig_r))
        
    
    
    def plot(self,X_test,clf,title,number_plot):
        res = 0.01
        self.my_plt.subplot(self.grid_r,self.grid_c,number_plot+1)
        self.my_plt.title(title)
        self.my_plt.grid(True)
        
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),np.arange(y_min, y_max, res))
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        
     #   self.my_plot.contourf(xx, yy, Z, alpha=0.2, color=cmap_light)
        self.my_plt.pcolormesh(xx, yy, Z,alpha=0.2, cmap=cmap_light)
        self.my_plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=cmap_light)

        
    def save(self,title):
        self.my_plt.savefig("/Users/marcotreglia/.bin/ML/Homework4/plot/"+title+".png",dpi=100 )
        
        
        
    def show(self):
        self.my_plt.show()
        
        
def plot_data(x_tra,x_tes,x_val,y_tra,y_tes,y_val):
    plt.figure(figsize=(14,3))
    
    plt.subplot(1,3,1)
    plt.title("Data train")
    plt.grid()
    plt.scatter(x_tra[:,0],x_tra[:,1],c = y_tra,cmap= cmap_light)
       
    plt.subplot(1,3,2)
    plt.title("Data validation")
    plt.grid()
    plt.scatter(x_val[:,0],x_val[:,1],c = y_val,cmap= cmap_light)
    
    plt.subplot(1,3,3)
    plt.title("Data test")
    plt.grid()
    plt.scatter(x_tes[:,0],x_tes[:,1],c = y_tes,cmap= cmap_light)
    
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework4/plot/data.png",dpi=100 )
    plt.show()

def plot(X_test,clf,title):
        res = 0.01
        plt.title(title)
        plt.grid(True)
        y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
        x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, res),np.arange(y_min, y_max, res))
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z,alpha=0.2, cmap=cmap_light)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=cmap_light)
        plt.savefig("/Users/marcotreglia/.bin/ML/Homework4/plot/"+title+".png",dpi=100 )
        plt.show()
        
def plot_accuracy(accuracy):
    plt.title("Accurancy on the validation set")    
    plt.plot(np.asarray(accuracy))
    plt.savefig("/Users/marcotreglia/.bin/ML/Homework4/plot/accuracy.png",dpi=100 )
    plt.show()
    
        
