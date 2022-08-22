import numpy as np
from sklearn.metrics import r2_score

class ridge():
    def __init__(self):
        pass

    def prepare_data(self):
        data = np.genfromtxt('/Users/imchengliang/Downloads/Code/ML/Regulation/data.dat', delimiter = ',')
        x = data[:,0:100]
        y = data[:,100].reshape(-1, 1)
        X = np.column_stack((np.ones((x.shape[0], 1)), x))
        X_train, y_train = X[:70], y[:70]
        X_test, y_test = X[70:], y[70:]
        return X_train, y_train, X_test, y_test

    def initialize(self, dims):
        w = np.zeros((dims, 1)) 
        b=0
        return w, b

    def l2_loss(self, X, y, w, b, alpha):
        num_train = X.shape[0]
        num_feature = X.shape[1]
        # model function
        y_hat = np.dot(X, w) + b
        # loss function
        loss = np.sum((y_hat-y)**2)/num_train + alpha*np.sum(np.square(w))
        # partial derivatives of parameters
        dw = np.dot(X.T, (y_hat-y))/num_train + 2*alpha*w
        db = np.sum((y_hat-y)) / num_train
        return y_hat, loss, dw, db

    def ridge_train(self, X, y, learning_rate=0.01, epochs=300):
        w, b = self.initialize(X.shape[1])
        loss_list = []
        for i in range(1, epochs):
            # calculate the current estimation, loss, and partial derivatives
            y_hat, loss, dw, db = self.l2_loss(X, y, w ,b, 0.1)
            loss_list.append(loss)
            # update parameters based on gradient descent
            w += -learning_rate * dw
            b += -learning_rate * db

            if i % 500 == 0:
                print('epoch %d loss %f' % (i, loss))

            # save parameters
            params = {'w':w, 'b':b}
            # save gradient
            grads = {'dw':dw, 'db':db}

        return loss_list, loss, params, grads

    def predict(self, X, params): 
        w = params['w']
        b = params['b']
        y_pred = np.dot(X, w) + b 
        return y_pred

if __name__ == '__main__': 
    ridge = ridge()
    X_train, y_train, X_test, y_test = ridge.prepare_data()
    loss_list, loss, params, grads = ridge.ridge_train(X_train, y_train, 0.01, 3000) 
    print(params)
    y_pred = ridge.predict(X_test, params)
    print(r2_score(y_test, y_pred))