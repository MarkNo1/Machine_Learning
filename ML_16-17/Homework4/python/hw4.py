#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:56:27 2016

@author: markno1
"""

import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
from plot import plotting_grid,plot_data,plot,plot_accuracy
from tabulate import tabulate

#       LINEAR SVM
###############################################################################
#Load data
iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target


#Splitting data
X_temp, X_test, y_temp, y_test = train_test_split(X,Y,test_size=0.3,random_state=70)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp,y_temp,test_size=0.14,random_state=66)
#Plotting data
plot_data(X_train,X_test,X_validation,y_train,y_test,y_validation)

# Define usefull quantities
accuracy_list = []
C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])

#Classification for C in [10^-3,10^3]
classificators = []
#Plotter 
my_plt = plotting_grid(fig_r=12,fig_c=11,grid_r=4,grid_c=2)

for j, counter in enumerate(C_value):
    clf = svm.SVC(kernel='linear', C=counter).fit(X_train,y_train)
    accuracy_list.append(clf.score(X_validation,y_validation))
    classificators.append(clf)
    my_plt.plot(X_validation,clf,"Classification for C = {}".format(counter),j)
   
my_plt.save("classification.png")
my_plt.show()

plot_accuracy(accuracy_list)
print(tabulate([accuracy_list]))


#best_C=C_value[]
for j, c in enumerate(C_value):
    title = "Best classification with C = {}".format(c)
    clf_best = svm.SVC(kernel='linear',C=counter).fit(X_train,y_train)
    plot(X_validation,clf_best,title)
    print("\n\nAccuracy on test data : "+str(clf_best.score(X_test,y_test)*100)+" % \n")


#    NON  LINEAR SVM - RBF
###############################################################################
# Define accuracy
accuracy_list = []
#Classification for C in [10^-3,10^3]
classificators = []
#Plotter 

my_plt = plotting_grid(fig_r=12,fig_c=9,grid_r=4,grid_c=2)
accuracy_list = []

## Train the model with C parameter
for j, c in enumerate(C_value):
   clf = svm.SVC(kernel='rbf', C=c).fit(X_train,y_train)
   accuracy_list.append(clf.score(X_validation,y_validation))
   classificators.append(clf)
   my_plt.plot(X_validation,clf,"Classification for C = {}".format(c),j)
my_plt.save("train_non_lin_c_var.png")
my_plt.show()
plot_accuracy(accuracy_list)
print(tabulate([accuracy_list]))





#Plotter 
gamma_value = np.array([10**-2,10**-1,1,10,10**2])
accuracy_table = np.zeros([len(gamma_value)+1,len(C_value)+1])
## Train the model with C,gamma parameters
## Test on validation
for j, c in enumerate(C_value):
    my_plt = plotting_grid(fig_r=9,fig_c=12,grid_r=3,grid_c=2)
    accuracy_list = []
    accuracy_table[0,j+1]=c
    for i, gam in enumerate(gamma_value):
        accuracy_table[i+1,0] = gam
        clf = svm.SVC(kernel='rbf', C=c,gamma=gam).fit(X_train,y_train)
        accuracy_table[i+1,j+1]= clf.score(X_validation,y_validation)
        my_plt.plot(X_validation,clf,"Classification for C = {} and Gamma = {}".format(c,gam),i)
    title = "plot_val(C = "+str(c)+").png"
    my_plt.save(title)
    my_plt.show()

print(tabulate(accuracy_table.tolist(),tablefmt="latex"))


## Bestvalue of C and Gamma
## Test the model with C,gamma parameters
## Test on test
accuracy_table = np.zeros([len(gamma_value)+1,len(C_value)+1])

for j, c in enumerate(C_value):
    my_plt = plotting_grid(fig_r=9,fig_c=12,grid_r=3,grid_c=2)
    accuracy_list = []
    accuracy_table[0,j+1]=c
    for i, gam in enumerate(gamma_value):
        accuracy_table[i+1,0] = gam
        title = "Best classification with C = {}  and Gamma = {}".format(c,gam)
        clf = svm.SVC(kernel='rbf',C=c,gamma=gam).fit(X_train,y_train)
        accuracy_table[i+1,j+1]= clf.score(X_test,y_test)
        my_plt.plot(X_test,clf,"Classification for C = {} and Gamma = {} -T".format(c,gam),i)
    my_plt.save("plot_test(C = "+str(c)+").png")
    my_plt.show()
print(tabulate(accuracy_table.tolist(),tablefmt="latex"))


#Best Gamma=0.01 | C= 10
clf = clf = svm.SVC(kernel='rbf',C=10,gamma=0.01).fit(X_train,y_train)
plot(X_test,clf,"Best Classification with C = {} and Gamma = {}".format(10,0.01))
print(clf.score(X_test,y_test))



# K - FOLD
###############################################################################
from sklearn.model_selection import KFold
#Merge train with validation
x_tra = X_train.tolist() + X_validation.tolist()
y_tra = y_train.tolist() + y_validation.tolist()

x_tra = np.asarray(x_tra)
y_tra = np.asarray(y_tra)





## K Fold for 5 
kf = KFold(n_splits=5,random_state=None, shuffle=False)



accuracy_table = np.zeros([len(gamma_value)+1,len(C_value)+1])
c_list  = []
info_list = []
accuracy_table = np.zeros([len(gamma_value)+1,len(C_value)+1])

for j, c in enumerate(C_value):
    my_plt = plotting_grid(fig_r=9,fig_c=12,grid_r=3,grid_c=2)
    accuracy_table[0,j+1]=c
    for i, gam in enumerate(gamma_value):
        accuracy_table[i+1,0] = gam
        kfold_score = []
        for i_train,i_test  in kf.split(x_tra,y_tra):
            x_tr , y_tr = x_tra[i_train],y_tra[i_train]
            x_te , y_te = x_tra[i_test], y_tra[i_test]
            clf = svm.SVC(kernel='rbf',C=c,gamma=gam).fit(x_tr,y_tr)
            kfold_score.append(clf.score(x_te,y_te))
        accuracy_table[i+1,j+1]=max(kfold_score)


    
print(tabulate(accuracy_table.tolist(),tablefmt="latex"))





