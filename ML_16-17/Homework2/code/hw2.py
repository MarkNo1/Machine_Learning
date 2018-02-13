import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

colors = plt.cm.cool


def error_sum(pred, y_test):
    e = []
    for i in range(len(pred)):
        e.append((y_test[i] - pred[i])**2)
    return np.asarray(e).sum()


X_train = np.load("regression/regression_Xtrain.npy")
y_train = np.load("regression/regression_ytrain.npy")

X_test = np.load("regression/regression_Xtest.npy")
y_test = np.load("regression/regression_ytest.npy")

Linear_regression = LinearRegression()
Linear_regression.fit(X_train.reshape(-1, 1), y_train)

prediction = Linear_regression.predict(X_test.reshape(-1, 1))

plt.title("Data Visualization")
plt.scatter(X_train, y_train, label="Train Data", c="red", cmap=colors)
plt.scatter(X_test, y_test, label="Test Data", c="blue", cmap=colors)
plt.legend()
plt.savefig("../plot/00.png")
plt.show()
print("MSE : " + str(mean_squared_error(y_test, prediction)))


plt.title("Linear regression")
plt.plot(X_test, prediction, label="Model Prediction")
plt.scatter(X_test, y_test, label="Training Data")
plt.legend()
plt.savefig("../plot/plot/0.png")
plt.show()

mean_square_error = np.zeros((9, 1))


x_range = np.linspace(-1, 5.5, 50).reshape(-1, 1)
mean_square_error[0] = mean_squared_error(y_test, prediction)
list_square_prediction = []
plt.figure(figsize=(16, 12))
for j in range(1, 10):

    poly = PolynomialFeatures(degree=j, include_bias=False)

    polynomial_X_train = poly.fit_transform(X_train.reshape(-1, 1))

    Polynomial_Regression = LinearRegression()

    Polynomial_Regression.fit(polynomial_X_train, y_train)

    polynomial_test = poly.fit_transform(X_test.reshape(-1, 1))
    prediction_poly = Polynomial_Regression.predict(polynomial_test)

    plt.subplot(4, 3, j)
    plt.title("Polynomial degree " + str(j) + " ")
    plt.plot(X_test, prediction_poly, label="Model prediction")
    plt.scatter(X_test, y_test, c='r', label="Training Data")
    mean_square_error[j - 1] = mean_squared_error(y_test, prediction_poly)
    #print("Squared prediction errors for all n data points : "+str(error_sum(prediction_poly,y_test)))
    list_square_prediction.append(error_sum(prediction_poly, y_test))

plt.legend()
plt.savefig("../plot/2.png")
plt.show()

plt.title("Mean Square error over polynomial degree")
plt.plot(np.linspace(1, 9, 9).reshape(-1, 1), mean_square_error)
plt.savefig("../plot/3.png")
plt.show()

# for i in range(len(list_square_prediction)):
#    print("Polynomial degree "+str(i+1)+" - Squared prediction error : "+str(list_square_prediction[i]))
for i in range(len(mean_square_error)):
    print("Polynomial degree " + str(i + 1) +
          " - Mean Squared prediction error : " + str(mean_square_error[i]))
