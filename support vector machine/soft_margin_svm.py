import numpy as np
from numpy import linalg
from sklearn.metrics import accuracy_score
import cvxopt
import cvxopt.solvers
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

def gen_non_lin_separable_data():
    mean1, mean2 = np.array([0, 2]), np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1

    X_train = np.vstack((X1[:80], X2[:80]))
    y_train = np.hstack((y1[:80], y2[:80]))
    X_test = np.vstack((X1[80:], X2[80:]))
    y_test = np.hstack((y1[80:], y2[80:]))
    
    return X_train, y_train, X_test, y_test

def plot_classifier(X1_train, X2_train, clf):
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(clf.sv_x[:,0], clf.sv_x[:,1], s=100, c="g", label="support vector")

    X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.legend()
    plt.show()

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

class soft_margin_svm:
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = linear_kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K) 
        q = cvxopt.matrix(np.ones(n_samples) * -1) 
        A = cvxopt.matrix(y, (1, n_samples)) 
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -+ np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv_x[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_pred = np.zeros(len(X))
            for n in range(len(X)):
                s = 0
                for a, sv_y, sv_x in zip(self.a, self.sv_y, self.sv_x):
                    s += a * sv_y * np.dot(X[n], sv_x)
                y_pred[n] = s
            return y_pred + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__=="__main__":
    X_train, y_train, X_test, y_test = gen_non_lin_separable_data()
    clf = soft_margin_svm(C=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of soft margin svm based on sklearn: ', accuracy_score(y_test, y_pred))
    plot_classifier(X_train[y_train==1], X_train[y_train==-1], clf)
