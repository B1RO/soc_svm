from enum import Enum

import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets


from svm_metrics import accuracy_score, balanced_accuracy_score, true_positives, false_positives, true_negatives, \
    false_negatives


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


def append_bias_column(X, beta):
    n, m = X.shape
    biasColumn = np.ones((n, 1)) * beta
    return np.hstack((X, biasColumn))


def linear_kernel_implicit(x, y):
    return np.dot(x, y)


def polynomial_kernel_implicit(x, y, coef0, degree):
    return (np.dot(x, y) + coef0) ** degree


def gaussian_kernel_implicit(x, y, sigma):
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
        self._X = None
        self._y = None
        self._loss_type = LossType.l2
        self._sv = None
        self._C = 0.1
        self._no_bias = False
        self._beta = 1

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
    def beta(self):
        return self._beta

    @classifier_type.setter
    def beta(self, value):
        self._beta = value

    @property
    def no_bias(self):
        return self._no_bias

    @no_bias.setter
    def no_bias(self, value):
        self._no_bias = value

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
            H = (np.dot(Y.T, np.dot(self._kernel(X, X.T), Y)) + self._C ** (-1) * np.eye(y.shape[0]))
        elif self.loss_type == LossType.l1 and self.classifier_type == ClassifierType.SOFT_MARGIN:
            H = np.dot(Y.T, np.dot(self._kernel(X, X.T), Y))
            H = H + (1e-10 * np.diag(H))
        elif self.classifier_type == ClassifierType.HARD_MARGIN:
            H = np.dot(Y.T, np.dot(self._kernel(X, X.T), Y))
        H = cvxopt.matrix(H)
        return H

    def setupOptimization(self):
        n_samples, n_features = self._X.shape

        print(X.shape)
        P = self.buildHessian(self._X, self._y)

        # RHS
        q = cvxopt.matrix(np.ones(n_samples) * -1)

        # Equality constraint
        A = cvxopt.matrix(self._y, (1, n_samples))
        b: matrix = cvxopt.matrix(0.0)

        # Inequality constraint
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        return P, q, G, h, A, b

    def train(self, X, y):
        if self._no_bias:
            self._X = append_bias_column(X,self._beta)
        else:
            self._X = X
        self._y = y

        # Solve QP problem
        solution = cvxopt.solvers.qp(*self.setupOptimization())

        # Lagrange multipliers
        self.lagrange_multipliers = np.ravel(solution['x'])
        # Support vectors
        if self._classifier_type == ClassifierType.SOFT_MARGIN and self.loss_type == LossType.l1:
            self._sv = np.logical_and(self.lagrange_multipliers > 1e-5, self.lagrange_multipliers <= self.C)
        else:
            self._sv = self.lagrange_multipliers > 1e-5

    def predict(self, X):
        y = np.diag(self._y)
        if self.no_bias:
            XwithBias = append_bias_column(X,self._beta)
            return np.sign(np.dot(self._kernel(XwithBias, self._X.T), np.dot(y, self.lagrange_multipliers)))
        else:
            b = np.sum(
                np.dot(self._kernel(self._X[self._sv], self._X.T), np.dot(y, self.lagrange_multipliers)) -
                self._y[self._sv]) / self._sv.sum()

            return np.sign(np.dot(self._kernel(X, self._X.T), np.dot(y, self.lagrange_multipliers))) + b



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
        print("plot2")
        if(self.no_bias):
            X = append_bias_column(X,self._beta)
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
    svm.no_bias = True
    svm.set_kernel(KernelType.SIGMOID, coef0=-10, kappa=1)
    svm.loss_type = LossType.l2
    X, y = loadData("data/data_medium.training")

    svm.train(X, y)
    svm.plot2(X, y)

    predicted = svm.predict(X)
    print("accuracy", accuracy_score(y,predicted))
    print("balanced_accuracy", balanced_accuracy_score(y,predicted))
    print("TP", true_positives(y,predicted), "FP", false_positives(y,predicted), "TN", true_negatives(y,predicted), "FN", false_negatives(y,predicted) )
    print("Predicted:", predicted)
