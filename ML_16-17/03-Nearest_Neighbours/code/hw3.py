from sklearn import neighbors, datasets
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from sklearn.decomposition import PCA
import numpy as np
from lib3 import plot
from lib3 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate

print("ML - Homework3 - Marco Treglia\n")
colors = plt.cm.cool

# 1
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2
X = preprocessing.scale(X)
pca = PCA(n_components=2)
X_t = pca.fit_transform(X)

# 3
X_train, X_test, y_train, y_test = train_test_split(
    X_t, y, test_size=0.40, random_state=100)


# 4
accuracy_list = []
plt.figure(figsize=(12, 9))
for k in range(1, 11):
    title = "(k= " + str(k) + ")"
    print(title)
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    plt.subplot(3, 4, k)
    res = 0.01
    plt.title(title)
    plt.grid(True)
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, color=colors)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=colors)

    accuracy_list.append([k, clf.score(X_test, y_test) * 100])

plt.savefig("../plot/" + title + ".png", dpi=100)
plt.show()
print(accuracy_list)


accuracy_list = np.asarray(accuracy_list)
plt.title("Accuracy K 1 to 10")
plt.xlabel("K")
plt.plot(accuracy_list[:, 0], accuracy_list[:, 1], c='blue')
plt.savefig("../plot/" + "accuracy" + ".png", dpi=100)
plt.show()


# 6
accuracy_list = []
title = "\n3-Class classification (k= 3 weights = uniform)"
clf = neighbors.KNeighborsClassifier(n_neighbors=3,  weights='uniform')
clf.fit(X_train, y_train)
plot(X, X_test, X_train, clf, title)
accuracy_list.append(['Uniform', clf.score(X_test, y_test)])

title = "\n3-Class classification (k= 3 weights = distance)"
clf = neighbors.KNeighborsClassifier(n_neighbors=3,  weights='distance')
clf.fit(X_train, y_train)
plot(X, X_test, X_train, clf, title)
accuracy_list.append(['Distance', clf.score(X_test, y_test)])
print(accuracy_list)


# Plotting the gaus Fuction
functions = [my_func, my_func1, my_func2, my_func3]
color = ['red', 'blue', 'green', 'black']
label = ['alfa=0.1', 'alfa=10', 'alfa=100', 'alfa=1000']
x = np.linspace(0, 10, 50)
info1 = []
info1.append(['weight', 'distance'])
info2 = []
info2.append(['weight', 'distance'])
info3 = []
info3.append(['weight', 'distance'])
info4 = []
info4.append(['weight', 'distance'])

info_list = [info1, info2, info3, info4]

for j in range(4):
    y = []
    for i in range(len(x)):
        y.append(functions[j](x[i]))
        info_list[j].append([j, x[i], functions[j](x[i])])

    plt.plot(x, np.asarray(y), c=color[j], label=label[j])
    # print(tabulate(info_list[j]))
    plt.legend()
    plt.savefig("../plot/" + "gauss" + ".png", dpi=200)
    plt.show()


#7 - 8
accuracy_list = []
func_title = ["my_fuction, alfa=0.1", "my_fuction, alfa=10",
              "my_fuction, alfa=100", "my_fuction, alfa=1000"]
plt.figure(figsize=(12, 10))

for k in range(len(functions)):
    title = "(k= 3, weights = " + func_title[k] + " )"
    clf = neighbors.KNeighborsClassifier(n_neighbors=10,  weights=functions[k])
    clf.fit(X_train, y_train)
    res = 0.01

    plt.subplot(2, 2, k + 1)
    plt.title(title)
    plt.grid(True)

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))
    Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, color=colors)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=clf.predict(X_test), cmap=colors)
    accuracy_list.append([k, clf.score(X_test, y_test)])

plt.savefig("../plot/" + title + ".png", dpi=100)
plt.show()

print(tabulate(accuracy_list))
accuracy_list = np.asarray(accuracy_list)
plt.title("Accuracy incrementing alfa from 0.1 to 1000")
plt.plot(accuracy_list[:, 0] + 1, accuracy_list[:, 1], c='blue')
plt.savefig("../plot/" + "accuracyMyFunc" + ".png", dpi=100)
plt.show()


print("Accuracy : " + str(clf.score(X_test, y_test)) + ". \n")

# 10

param_grid = {'n_neighbors': np.arange(1, 11),
              'weights': ['uniform', 'distance', my_func, my_func1, my_func2, my_func3],
              'metric': ['euclidean', 'manhattan']
              }

KN = neighbors.KNeighborsClassifier()
clf_grid = GridSearchCV(KN, param_grid)
clf_grid.fit(X_train, y_train)
predict = clf_grid.predict(X_test)
acc = clf_grid.best_estimator_.score(X_test, y_test)

plot(X, X_test, X_train, clf_grid.best_estimator_, "Best Estimator")

print("Accuracy = " + str(acc))

accuracy_table = []
n_neigh = np.arange(1, 11)
weight = ['uniform', 'distance', my_func, my_func1, my_func2, my_func3]
metrics = ['euclidean', 'manhattan']
accuracy_table.append(["METRIC", "WEIGHT", "N-NEIGHT", "ACCURACY"])

for i in range(len(metrics)):
    for j in range(len(weight)):
        for k in range(len(n_neigh)):
            title = "KN - Metric : {} | Weight : {} | N_neightbors : {} ".format(
                metrics[i], weight[j], n_neigh[k])
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=n_neigh[k],  weights=weight[j], metric=metrics[i])
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            accuracy_table.append([metrics[i], weight[j], n_neigh[k], accuracy])

# print(tabulate(accuracy_table, tablefmt='latex', floatfmt='.2f'))

title = "KN - Metric : manhattan | Weight : my_func | N_neightbors : 9 "
clf = neighbors.KNeighborsClassifier(n_neighbors=9,  weights=my_func, metric='manhattan')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
plot(X, X_test, X_train, clf, title)
