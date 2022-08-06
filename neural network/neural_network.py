import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Neural_network:
    def __init__(self):
        pass

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def layer_sizes(self, X, Y):
        # input layer
        n_x = X.shape[0] 
        # hidden layer
        n_h = 4
        # output layer
        n_y = Y.shape[0] 
        return (n_x, n_h, n_y)

    def initialize_parameters(self, n_x, n_h, n_y):
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))

        assert (W1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (W2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def forward_propagation(self, X, parameters):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        Z1 = np.dot(W1, X) + b1 
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2 
        A2 = self.sigmoid(Z2)
        assert(A2.shape == (1, X.shape[1]))

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def compute_cost(self, A2, Y, parameters):
        # training sample size
        m = Y.shape[1]
        # compute cross-entropy loss
        logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2), 1-Y)
        cost = -1/m * np.sum(logprobs)
        # compress dimension
        cost = np.squeeze(cost)

        assert(isinstance(cost, float))
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        m = X.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2-Y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True) 
        dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2)) 
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
        
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}    
        return grads

    def update_parameters(self, parameters, grads, learning_rate=1.2):
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        W1 -= dW1 * learning_rate
        b1 -= db1 * learning_rate
        W2 -= dW2 * learning_rate
        b2 -= db2 * learning_rate

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        return parameters

    def nn_model(self, X, Y, n_h, num_iterations=10000, print_cost=False):
        np.random.seed(3)
        n_x = self.layer_sizes(X, Y)[0]
        n_y = self.layer_sizes(X, Y)[2]
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        # loop for gradient descent and parameter update
        for i in range(0, num_iterations):
            # forward propagation
            A2, cache = self.forward_propagation(X, parameters)
            # compute current loss
            cost = self.compute_cost(A2, Y, parameters)
            # backward propagation
            grads = self.backward_propagation(parameters, cache, X, Y)
            # update parameters
            parameters = self.update_parameters(parameters, grads, learning_rate=1.2)
            # print loss
            if print_cost and i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        return parameters

    def predict(self, parameters, X):
        A2, cache = self.forward_propagation(X, parameters) 
        predictions = (A2>0.5)
        return predictions

def create_dataset():
    np.random.seed(1)
    # data size
    m = 400
    # number of instance for every label
    N = int(m/2)
    # data dimension
    D = 2
    # data matrix
    X = np.zeros((m, D))
    # label dimension
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N*j, N*(j+1))
        # theta
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2  
        # radius
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T
    return X, Y

if __name__ == "__main__":
    model = Neural_network()
    X, Y = create_dataset()
    parameters = model.nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
    predictions = model.predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')