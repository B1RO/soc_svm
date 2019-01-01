import sklearn
from cvxopt.base import matrix
from sklearn import datasets

import numpy as np
import cvxopt
import cvxopt.solvers

import matplotlib.pyplot as plt


def buildHessian(X, y):
    # Gram matrix
    K = np.dot(X, X.T)
    K = cvxopt.matrix(K)

    # Hessian matrix
    H = np.outer(y, y.T) * K
    H = cvxopt.matrix(H)
    return H


def buildHessian2(X, y):
    Y = np.diag(y)
    H = np.dot(np.dot(np.dot(Y, X), X.T), Y)
    H = cvxopt.matrix(H)
    return H


class SVM(object):
    def buildHessian(X, y):
        Y = np.diag(y)
        H = np.dot(np.dot(np.dot(Y, X), X.T), Y);

    def train(self, X, y):
        n_samples, n_features = X.shape

        P = buildHessian(X, y)
        # P = buildHessian2(X, y)

        # RHS
        q = cvxopt.matrix(np.ones(n_samples) * -1)

        # Equality constraint
        A = cvxopt.matrix(y, (1, n_samples))
        b: matrix = cvxopt.matrix(0.0)

        # Inequality constraint
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors
        sv = a > 1e-5

        # TODO reconstruction formulas

        # normal of separating hyperplane
        # algorithmically
        normal = np.sum([a[i]*y[i]*X[i] for i in range(0, len(X))],axis=0)
        # using matrix algebra
        #normal = np.dot(np.dot(np.diag(a), X).T, y)

        indexset = [index for index, value in enumerate(sv) if value]
        shift = 1 / (len(indexset)) * np.sum([np.dot(normal, X[i].T) - y[i] for i in indexset])
        self.normal = normal
        self.shift = shift
        return shift, normal

    # TODO implement predict
    def predict(self, X):
        return (np.dot(self.normal,X.T) - self.shift) >= 0


def loadData(file):
    data = sklearn.datasets.load_svmlight_file(file);
    return np.array(data[0].todense()), data[1]


def plotData(X, y):
    class1 = y > 0
    X1 = X[class1]
    X2 = X[~class1]
    plt.plot(X1[:, 0], X1[:, 1], "ro")
    plt.plot(X2[:, 0], X2[:, 1], "bo")
    plt.show()


if __name__ == "__main__":
    svm = SVM()

    X, y = loadData("data/small")

    w, b = svm.train(X, y)
    #plotData(X, y)


    for a,b in zip(X,y):
        print(svm.predict(a))
