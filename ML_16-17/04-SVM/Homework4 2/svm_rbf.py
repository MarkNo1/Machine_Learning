import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap
from plot import plot
from tabulate import 

#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_temp, X_test, y_temp, y_test = train_test_split(X,Y,test_size=0.3,random_state=70)
X_train, X_validation, y_train, y_validation = train_test_split(X_temp,y_temp,test_size=0.14,random_state=66)

#Data
cmap_light = (ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA']))


# Define usefull quantities

accuracy_vector=np.zeros((7,1))
C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
h=0.01

#Classification for C in [10^-3,10^3]
for j, counter in enumerate(C_value):
    clf = svm.SVC(kernel='rbf', C=counter).fit(X_train,y_train)
    accuracy_vector[j]=clf.score(X_validation,y_validation)
    plot(X_validation,clf,"Classification for C = {}".format(counter))
    

plt.plot(accuracy_vector)
plt.show()

print("\n\n"+str(accuracy_vector))

best_C=C_value[3]

clf_best = svm.SVC(kernel='rbf',C=best_C).fit(X_train,y_train)
prediction_test=clf_best.predict(X_test)
plot(X_test,clf_best,"Best Classification with C = {}".format(counter))
print("\n\nAccuracy on test data : "+str(clf_best.score(X_test,y_test)*100)+" % \n")