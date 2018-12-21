import sklearn
from sklearn import datasets

import numpy as np
import cvxopt
import cvxopt.solvers

import matplotlib.pyplot as plt

#class SVM(object):

    #def train(self, X, y):

    #def predict(self, X, y):

def loadData(file):
    data = sklearn.datasets.load_svmlight_file(file);
    return np.array(data[0].todense()), data[1]

def plotData(X, y):
    class1 = y > 0
    X1 = X[class1]
    X2 = X[~class1]
    plt.plot(X1[:,0], X1[:,1], "ro")
    plt.plot(X2[:,0], X2[:,1], "bo")
    plt.show()

if __name__ == "__main__":
    #svm = SVM()

    X, y = loadData("data/small")
    #plotData(X, y)
