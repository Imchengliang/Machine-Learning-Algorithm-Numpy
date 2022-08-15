import numpy as np
from numpy import linalg
from sklearn.metrics import accuracy_score
import cvxopt
import cvxopt.solvers
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

def gen_non_linear_sep_data():
    mean1, mean2, mean3, mean4 = [-1, 2], [1, -1], [4, -4], [-4, 4]
    cov = [[1.0, 0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = -1 * np.ones(len(X2))
    X_train = np.vstack((X1[:80], X2[:80]))
    y_train = np.hstack((y1[:80], y2[:80]))
    X_test = np.vstack((X1[80:], X2[80:]))
    y_test = np.hstack((y1[80:], y2[80:]))

    return X_train, y_train, X_test, y_test

def plot_classifier(X1_train, X2_train, clf):
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "go")
    plt.scatter(clf.sv_x[:,0], clf.sv_x[:,1], s=100, c="blue", edgecolors="b", label="support vector")

    X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
    plt.legend()
    plt.show()

def polynomial_kernel(x, y, p=3):
    return (1+np.dot(x, y)) ** p

class non_linear_svm:
    def __init__(self, kernel=polynomial_kernel):
        self.kernel = kernel

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = polynomial_kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K) 
        q = cvxopt.matrix(np.ones(n_samples) * -1) 
        A = cvxopt.matrix(y, (1, n_samples)) 
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h = cvxopt.matrix(np.zeros(n_samples))

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
        for i in range(len(self.a)):
            self.b += self.sv_y[i]
            self.b -+ np.sum(self.a * self.sv_y * K[ind[i], sv])
        self.b /= len(self.a)

        # Weight vector
        self.w = None

    def project(self, X):
        y_pred = np.zeros(len(X))
        for i in range(X.shape[0]):
            s = 0
            for n, spv_y, spv_x in zip(self.a, self.sv_y, self.sv_x):
                s += n * spv_y * polynomial_kernel(X[i], spv_x)
            y_pred[i] = s
            
        return y_pred + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__=="__main__":
    X_train, y_train, X_test, y_test = gen_non_linear_sep_data()
    clf = non_linear_svm()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy of soft margin svm based on sklearn: ', accuracy_score(y_test, y_pred))
    plot_classifier(X_train[y_train==1], X_train[y_train==-1], clf)