import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class Perceptron:
    def __init__(self):
        pass

    def sign(self, x, w, b):
        return np.dot(x,w)+b

    def initilize_with_zeros(self, dim):
        w = np.zeros(dim) 
        b = 0.0
        return w, b

    def train(self, X_train, y_train, learning_rate):
        w, b = self.initilize_with_zeros(X_train.shape[1])
        # initialize the misclassification
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for i in range(len(X_train)):
                X = X_train[i]
                y = y_train[i]
                # if there is misclassification, update the parameters until no misclassification exits
                if y * self.sign(X, w, b) <= 0:
                    w = w + learning_rate*np.dot(y, X) 
                    b = b + learning_rate*y 
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
                print('There is no missclassification!')

            # save the updated parameters
            params = {'w': w, 'b': b}

        return params

if __name__ == "__main__":
    model = Perceptron()

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:,:-1], data[:,-1]
    y = np.array([1 if i==1 else -1 for i in y])

    params = model.train(X, y, 0.01) 
    x_points = np.linspace(4, 7, 10)
    y_hat = -(params['w'][0]*x_points + params['b']) / params['w'][1]
    plt.plot(x_points, y_hat)

    plt.scatter(data[:50, 0], data[:50, 1], color='red', label='0')
    plt.scatter(data[50:100, 0], data[50:100, 1], color='green', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()