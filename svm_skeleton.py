import sklearn
from sklearn import datasets

import numpy as np
import cvxopt
import cvxopt.solvers

import matplotlib.pyplot as plt

class SVM(object):

    def train(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.dot(X,X.T);
        K = cvxopt.matrix(K);

        # Hessian
        P = np.outer(y, y.T) * K
        P = cvxopt.matrix(P);

        # RHS
        q = cvxopt.matrix(np.ones(n_samples) * -1)

        # Equality constraint
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)

        # Inequality constraint
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])
        print(a)

        # Support vectors
        sv = a > 1e-5
        print(sv)

        #TODO reconstruction formulas

    #TODO implement predict    
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
    svm = SVM()

    X, y = loadData("data/small")
    #plotData(X, y)

    svm.train(X, y)
