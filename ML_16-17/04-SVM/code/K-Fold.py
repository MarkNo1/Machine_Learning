import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets,svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from bottle import bottleneck3D

#Load data

iris = datasets.load_iris()
X=np.vstack((iris.data[:,0],iris.data[:,1])).T
Y=iris.target

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=50)

# Define usefull quantities

C_value = np.array([10**-3,10**-2,10**-1,1,10,10**2,10**3])
gamma_value = np.array([10**-2,10**-1,1,10,10**2])
h=0.01

scores = np.zeros((7,5,5))

for j, counter in enumerate(C_value):
    for i, counter2 in enumerate(gamma_value):
            clf = svm.SVC(kernel='rbf', gamma=counter2, C=counter).fit(X_train,y_train)
            prediction = clf.predict(X_test)
            scores[j,i,:]=cross_val_score(clf, X_train, y_train, cv=5)
            x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
            y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            cmap_light = (ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA']))
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
            plt.scatter(X_test[:,0], X_test[:,1] , c=prediction)
            plt.title('C=' +str(counter) + '   Gamma =' +str(counter2))
            plt.show()

#Check the relative value in the scores matrix before choosing the max
max_1,max_2,max_3 = bottleneck3D(scores)
            
best_C=C_value[max_1[0]]
best_gamma=gamma_value[max_1[1]]
clf = svm.SVC(kernel='rbf',gamma=best_gamma,C=best_C).fit(X_train,y_train)
prediction=clf.predict(X_test)
accuracy=clf.score(X_test,y_test)
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(['#AAAAFF','#AAFFAA','#FFAAAA'])
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_test[:,0], X_test[:,1] , c=prediction)
plt.title('C=' +str(best_C) + '   Gamma=' +str(best_gamma))
plt.show()
print('Accuracy: '+(str(accuracy*100)+ '%'))