import warnings
import seaborn as sns
import pandas as pd
from enum import Enum
import seaborn as sns
import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from colorama import init

import k_fold_cross_validation
from ClassifierType import ClassifierType
from hyperparameter_optimization.nested_grid_search import nested_grid_search
from hyperparameter_optimization.randomized_search import randomized_search, nested_randomized_search
from svm_metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, AUC_score, \
    specificity_score
from validation import train_test_split

init()

from KernelType import KernelType


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
    print(sigma)
    for i, vec1 in enumerate(x):
        for j, vec2 in enumerate(y.T):
            diff = vec1 - vec2
            P[i][j] = np.exp(-(np.dot(diff, diff)) / (2 * sigma ** 2))

    return P


class SVM(object):
    def __init__(self):
        self.lagrange_multipliers = []
        self._sv = []
        self.w = np.array([])
        self.b = 0
        self._kernel_type = KernelType.LINEAR
        self._classifier_type = ClassifierType.SOFT_MARGIN
        self._X = None
        self._y = None
        self._loss_type = LossType.l2
        self._sv = None
        self._C = 0.1
        self._no_bias = False
        self._beta = 1
        self._useDifferentErrorCosts = False
        self._coef0 = 0
        self._degree = 1
        self._sigma = 0

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value

    @property
    def useDifferentErrorCosts(self):
        return self._useDifferentErrorCosts

    @useDifferentErrorCosts.setter
    def useDifferentErrorCosts(self, value):
        self.C = [1, 1]
        self._useDifferentErrorCosts = value

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
    def coef0(self):
        return self._coef0

    @coef0.setter
    def coef0(self, value):
        self._coef0 = value

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        self._degree = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, value):
        self._kernel_type = value

    def _kernel(self, x, y):
        if self._kernel_type == KernelType.LINEAR:
            return linear_kernel_implicit(x, y)
        elif self._kernel_type == KernelType.GAUSSIAN:
            return gaussian_kernel_implicit(x, y, self._sigma)
        elif self._kernel_type == KernelType.POLYNOMIAL:
            return polynomial_kernel_implicit(x, y, self._coef0, self._degree)

    def buildHessian(self, X, y):
        Y = np.diag(y)
        if self.loss_type == LossType.l2:
            H = (np.dot(Y.T, np.dot(self._kernel(X, X.T), Y)) + (self._C ** (-1)) * np.eye(y.shape[0]))
        elif self.loss_type == LossType.l1 and self.classifier_type == ClassifierType.SOFT_MARGIN:
            H = np.dot(Y.T, np.dot(self._kernel(X, X.T), Y))
            H = H + (1e-10 * np.diag(H))
        elif self.classifier_type == ClassifierType.HARD_MARGIN:
            H = np.dot(Y.T, np.dot(self._kernel(X, X.T), Y))
        H = cvxopt.matrix(H)
        return H

    def setupOptimization(self):
        cvxopt.solvers.options['show_progress'] = False

        m, n = self._X.shape

        P = self.buildHessian(self._X, self._y)

        # RHS
        q = cvxopt.matrix(np.ones(m) * -1)

        # Equality constraint
        A = cvxopt.matrix(self._y, (1, m))
        b: matrix = cvxopt.matrix(0.0)

        q = cvxopt.matrix(-np.ones((m, 1)))

        # Inequality constraint
        if self.classifier_type == ClassifierType.SOFT_MARGIN and self.loss_type == LossType.l1:
            G = cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
            h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self._C)))
        else:
            G = cvxopt.matrix(np.diag(np.ones(m) * -1))
            h = cvxopt.matrix(np.zeros(m))

        return P, q, G, h, A, b

    def fit(self, X, y):
        if self._no_bias:
            self._X = append_bias_column(X, self._beta)
        else:
            self._X = X
        self._y = y

        # Solve QP problem
        try:
            solution = cvxopt.solvers.qp(*self.setupOptimization())

            # Lagrange multipliers
            self.lagrange_multipliers = np.ravel(solution['x'])

            if self.useDifferentErrorCosts:
                positiveSupportVectors = np.logical_and(self.lagrange_multipliers[self._y == 1] > 1e-5,
                                                        self.lagrange_multipliers[self._y == 1] <= self.C[0])
                negativeSupportVectors = np.logical_and(self.lagrange_multipliers[self._y == 1] > 1e-5,
                                                        self.lagrange_multipliers[self._y == -1] <= self.C[1])
                self._sv = np.concatenate(positiveSupportVectors, negativeSupportVectors)

            # Support vectors
            if self._classifier_type == ClassifierType.SOFT_MARGIN and self.loss_type == LossType.l1:
                self._sv = np.logical_and(self.lagrange_multipliers > 1e-5, self.lagrange_multipliers <= self.C)
            else:
                self._sv = self.lagrange_multipliers > 1e-5
        except ValueError:
            warnings.warn("The optimization problem has no solution")

    def predict_raw(self, X):
        y = np.diag(self._y)
        if hasattr(self._sv, "__len__") == False or len(self._sv) != len(self._y) or np.sum(self._sv) == 0:
            warnings.warn("Something went wrong with  support vectors, either the solver returned"
                          "an unexpected number of support vectors or no support vectors were found")
            return np.ones(len(X))
        if self.no_bias:
            XwithBias = append_bias_column(X, self._beta)
            return np.dot(self._kernel(XwithBias, self._X.T), np.dot(y, self.lagrange_multipliers))
        else:
            b = np.sum(self._y[self._sv] -
                       np.dot(self._kernel(self._X[self._sv], self._X.T),
                              np.dot(y, self.lagrange_multipliers))) / self._sv.sum()
            return np.dot(self._kernel(X, self._X.T), np.dot(y, self.lagrange_multipliers)) + b

    def predict(self, X):
        return np.sign(self.predict_raw(X))

    def view(self, X, y, resolution=200):
        print("plot2")
        if (self.no_bias):
            X = append_bias_column(X, self._beta)
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

        plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], "ko")
        plt.plot(X[y == 1 * self._sv][:, 0], X[y == 1 * self._sv][:, 1], "ko")
        plt.plot(X[y == -1][:, 0], X[y == -1][:, 1], "ko")
        plt.plot(X[y == -1 * self._sv][:, 0], X[y == -1 * self._sv][:, 1], "ko")
        print("rryy")
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
    svm.loss_type = LossType.l1
    svm.kernel_type = KernelType.POLYNOMIAL
    svm.no_bias = False

    datasets = ["abl1.svmlight", "adora2a.svmlight", "adora3.svmlight", "cnr1.svmlight", "cnr2.svmlight"];

    np.seterr(all='raise')
    scores = pd.DataFrame()
    results = []
    for dataset in datasets:
        X, y = loadData("data/" + dataset)
        X_train, y_train, X_test, y_test = train_test_split(X, y, 0.3)
        result, df = nested_grid_search(svm, [accuracy_score,balanced_accuracy_score,precision_score,recall_score,specificity_score,f1_score,AUC_score], X, y, 4, 4,
                                        C=[0.01,0.1, 0.25, 0.5, 1, 10, 100],
                                        degree=[1,2,3,4,5],
                                        coef0=[1,2,5,10,100,1000])

        scores = scores.append(pd.Series(result.score), ignore_index=True)
        df.to_pickle("./out/" + dataset.split('.')[0] + "_polynomial_l2_gridsearch")
        # result = nested_randomized_search(svm, [accuracy_score,balanced_accuracy_score,precision_score,recall_score,specificity_score,f1_score,AUC_score], X,y, 4, 50, {"C" : [0, 1000]})
        # plt.xlabel("iterations")
        # plt.ylabel("accuracy")
        # plt.title(dataset.split('.')[0])
        # plt.show()

    scores.to_pickle("./out/polynomial_gridsearch_l2_scores")
    # print(dataset.split('.'`)[0],result)`
    # sns.set(style="whitegrid")
    # df = pd.DataFrame(nested_grid_search.scores)
    # df = df.T
    # df.columns = [x.split('.')[0] for x in datasets]
    # df.to_pickle("l2gridsearch")
    # sns.swarmplot(data=df, color=".2").plot()
    # sns.boxplot(data=df, whis=np.inf).plot()
    # plt.title("Distribution of accuracy based on varying C parameter, L1 - bias")
    # plt.show()

