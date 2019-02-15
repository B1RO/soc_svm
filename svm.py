import sklearn
from cvxopt.base import matrix
from sklearn import datasets

import numpy as np
import cvxopt
import cvxopt.solvers

import matplotlib.pyplot as plt

class SVM(object):
    lagrange_multipliers = []
    w = np.array([])
    b = 0

    def buildHessian(self, X, y):
        Y = np.diag(y)
        H = np.dot(np.dot(np.dot(Y, X), X.T), Y)
        H = cvxopt.matrix(H)
        return H

    def train(self, X, y):
        n_samples, n_features = X.shape

        P = self.buildHessian(X, y)

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
        self.lagrange_multipliers = np.ravel(solution['x'])
        # Support vectors
        sv = self.lagrange_multipliers > 1e-5

        self.reconstruct(X, y)

    def predict(self, X):
        return np.sign(X @ self.w - self.b)

    def reconstruct(self, X, y):
        self.w = np.asarray(y @ np.diag(self.lagrange_multipliers) @ np.matrix(X)).flatten()
        sv = self.lagrange_multipliers > 1e-5
        self.b = ((X[sv] @ self.w.T) - y[sv].T).sum() / sv.sum()

    def plot(self, X, y):
        sv = self.lagrange_multipliers > 1e-5
        plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], "bo")
        plt.plot(X[y==1 * sv][:, 0], X[y==1 * sv][:, 1], "co", markersize=14)
        plt.plot(X[y == -1][:, 0], X[y == -1][:, 1], "ro")
        plt.plot(X[y == -1 * sv][:, 0], X[y == -1 * sv][:, 1], "mo", markersize=14)

        axes = plt.gca()
        ymin, ymax =  axes.get_ylim()
        linspace = np.linspace(ymin, ymax)
        line_y = (self.w[0] * linspace - self.b) / -self.w[1]
        support0_y = (self.w[0] * linspace - np.dot(X[y == 1 * sv][0],self.w)) / -self.w[1]
        support1_y = (self.w[0] * linspace - np.dot(X[y == -1 * sv][0], self.w)) / -self.w[1]
        plt.plot(linspace, line_y)
        plt.plot(linspace, support0_y, "c")
        plt.plot(linspace, support1_y, "m")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Plot of hyperplane separating 2 classes using SVM")

        plt.show()

    def accuracy_score(self, predicted, actual):
        return np.sum(predicted == actual)/len(actual)


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

    X, y = loadData("data/data_medium.training")

    svm.train(X, y)
    svm.plot(X, y)

    predicted = svm.predict(X)
    print(svm.accuracy_score(predicted, y))
    print("Predicted:", predicted)