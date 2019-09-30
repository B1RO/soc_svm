from logging import warning

import sklearn
from cvxopt.base import matrix
from sklearn import datasets

import numpy as np
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
from enum import Enum  # for enum34, or the stdlib version
import math


class ClassifierType(Enum):
    SOFT_MARGIN = "soft_margin"
    HARD_MARGIN = "hard_margin"


class KernelType(Enum):
    LINEAR = "linear"
    POLYNOMIAL = 'polynomial'
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"


class LossType(Enum):
    l1 = "l1"
    l2 = "l2"


def linear_kernel_implicit(x, y):
    return np.dot(x, y)


def polynomial_kernel_implicit(x, y, coef0, degree):
    return (np.dot(x, y) + coef0) ** degree


def gaussian_kernel_implicit(x, y, sigma):
    # TODO: Vectorize
    P = np.empty((len(x), len(y.T)))
    for i, vec1 in enumerate(x):
        for j, vec2 in enumerate(y.T):
            diff = vec1 - vec2
            P[i][j] = np.exp(-(np.dot(diff, diff)) / (2 * sigma ** 2))

    return P


def sigmoid_kernel_implicit(x, y, kappa, coef0):
    return np.tanh(kappa * np.dot(x, y) + coef0)


class SVM(object):
    def __init__(self):
        self.lagrange_multipliers = []
        self.w = np.array([])
        self.b = 0
        self.set_kernel(linear_kernel_implicit)
        self._classifier_type = ClassifierType.SOFT_MARGIN
        self._original_X = None
        self._original_y = None
        self._loss_type = LossType.l2
        self._sv = None
        self._C = 0.1

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def loss_type(self):
        return self._loss_type

    @loss_type.setter
    def loss_type(self, value):
        self._loss_type = value

    @property
    def classifier_type(self):
        return self._classifier_type

    @classifier_type.setter
    def classifier_type(self, value):
        self._classifier_type = value

    @property
    def kernel(self):
        return self._kernel

    def set_kernel(self, type, degree=None, gamma=None, coef0=None, kappa=None):
        if type == KernelType.LINEAR:
            self._kernel = linear_kernel_implicit
        if type == KernelType.POLYNOMIAL:
            self._kernel = lambda x, y: polynomial_kernel_implicit(x, y, coef0, degree)
        if type == KernelType.GAUSSIAN:
            self._kernel = lambda x, y: gaussian_kernel_implicit(x, y, gamma)
        if type == KernelType.SIGMOID:
            self._kernel = lambda x, y: sigmoid_kernel_implicit(x, y, kappa, coef0)

    def buildHessian(self, X, y):
        Y = np.diag(y)
        if self.loss_type == LossType.l2:
            H = (np.dot(Y.T, np.dot(self._kernel(X, X.T), Y)) + np.linalg.inv(np.dot(self._C, np.identity(y.shape[0]))))
        elif self.loss_type == LossType.l1 or self.classifier_type == ClassifierType.HARD_MARGIN:
            H = np.dot(Y.T, np.dot(self._kernel(X, X.T), Y))
            H = H + 1e-10
        H = cvxopt.matrix(H)
        return H

    def train(self, X, y):
        n_samples, n_features = X.shape
        self._original_X = X
        self._original_y = y

        print(X.shape)
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
        if self._classifier_type == ClassifierType.SOFT_MARGIN and self.loss_type == LossType.l1:
            self._sv = np.logical_and(self.lagrange_multipliers > 1e-5, self.lagrange_multipliers <= self.C)
        else:
            self._sv = self.lagrange_multipliers > 1e-5


    def predict(self, X):
        y = np.diag(self._original_y)
        b = np.sum(
            np.dot(self._kernel(self._original_X[self._sv], self._original_X.T), np.dot(y, self.lagrange_multipliers)) -
            self._original_y[self._sv]) / self._sv.sum()

        return np.sign(np.dot(self._kernel(X, self._original_X.T), np.dot(y, self.lagrange_multipliers))) + b

    def plot(self, X, y):
        sv = self.lagrange_multipliers > 1e-5
        plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], "bo")
        plt.plot(X[y == 1 * sv][:, 0], X[y == 1 * sv][:, 1], "co", markersize=14)
        plt.plot(X[y == -1][:, 0], X[y == -1][:, 1], "ro")
        plt.plot(X[y == -1 * sv][:, 0], X[y == -1 * sv][:, 1], "mo", markersize=14)

        axes = plt.gca()
        ymin, ymax = axes.get_ylim()
        linspace = np.linspace(ymin, ymax)
        line_y = (self.w[0] * linspace - self.b) / -self.w[1]
        support0_y = (self.w[0] * linspace - np.dot(X[y == 1 * sv][0], self.w)) / -self.w[1]
        support1_y = (self.w[0] * linspace - np.dot(X[y == -1 * sv][0], self.w)) / -self.w[1]
        plt.plot(linspace, line_y)
        plt.plot(linspace, support0_y, "c")
        plt.plot(linspace, support1_y, "m")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Plot of hyperplane separating 2 classes using SVM")
        plt.show()

    def plot2(self, X, y, resolution=100):
        x1_min = X[:, 0].min() - 1
        x1_max = X[:, 0].max() + 1
        x2_min = X[:, 1].min() - 1
        x2_max = X[:, 1].max() + 1

        xv, yv = np.meshgrid(np.linspace(x1_min, x1_max, resolution), np.linspace(x2_min, x2_max, resolution))

        Z = self.predict(np.array([xv.ravel(), yv.ravel()]).T)
        Z = Z.reshape(xv.shape)

        plt.contourf(xv, yv, Z, alpha=0.6)
        plt.xlim(xv.min(), xv.max())
        plt.ylim(yv.min(), yv.max())

        plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], "bo")
        plt.plot(X[y == 1 * self._sv][:, 0], X[y == 1 * self._sv][:, 1], "co", markersize=14)
        plt.plot(X[y == -1][:, 0], X[y == -1][:, 1], "ro")
        plt.plot(X[y == -1 * self._sv][:, 0], X[y == -1 * self._sv][:, 1], "mo", markersize=14)

        plt.show()

    def accuracy_score(self, predicted, actual):
        return np.sum(predicted == actual) / len(actual)


def loadData(file):
    data = sklearn.datasets.load_svmlight_file(file)
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
    svm.classifier_type = ClassifierType.SOFT_MARGIN
    svm.set_kernel(KernelType.SIGMOID, coef0=-10, kappa=1)
    svm.loss_type = LossType.l2
    X, y = loadData("data/data_medium.training")

    svm.train(X, y)
    svm.plot2(X, y)

    predicted = svm.predict(X)
    print(svm.accuracy_score(predicted, y))
    print("Predicted:", predicted)
