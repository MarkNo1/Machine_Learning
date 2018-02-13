from bottle import bottleneck2D
import numpy as np
from lib4 import plot_mesh_gamma
from lib4 import plot_density
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import lib4





#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_temp, X_test, y_temp, y_test = train_test_split(X,Y,test_size=0.3,random_state=69)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp,y_temp,test_size=0.14,random_state=69)

# Define usefull quantities

accuracy_vector=np.zeros((7,5))
C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
gamma_value = np.array([10**-2,10**-1,1,10,10**2])
h=0.01

classifiers = []

for j, c in enumerate(C_value):
    lib4.pltt.figure(figsize=(20,12))
    for i, gamma in enumerate(gamma_value):
        clf = svm.SVC(kernel='rbf',gamma=gamma, C=c).fit(X_train,y_train)
        accuracy_vector[j,i]=clf.score(X_validation,y_validation)
        lib4.pltt.subplot(len(C_value),len(gamma_value),i+1)
        plot_mesh_gamma(X_validation,clf,c,gamma)
        classifiers.append((c,gamma,clf))
    lib4.pltt.show
	  
        
        
#Evaluate the best parameters
#Check the relative value in the accurancy_vector before choosing the max

max_1,max_2,max_3 = bottleneck2D(accuracy_vector)

best_C=C_value[max_2[0]]
best_gamma=gamma_value[max_2[1]]
h=0.01
clf = svm.SVC(kernel='rbf',gamma=best_C, C=best_gamma).fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
plot_mesh_gamma(X_temp,clf,best_C,best_gamma)
print('Accuracy: '+(str(accuracy*100)+ '%'))



h=0.01
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X, Y)

