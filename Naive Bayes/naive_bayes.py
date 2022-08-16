import numpy as np
import pandas as pd

class naive_bayes:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = y[y.columns[0]].unique() 
        class_count = y[y.columns[0]].value_counts() 
        class_prior = class_count/len(y)

        prior = dict()
        for col in X.columns:
            for j in classes:
                p_x_y = X[(y==j).values][col].value_counts()
                for i in p_x_y.index:
                    prior[(col, i, j)] = p_x_y[i]/class_count[j]

        return classes, class_prior, prior

    def predict(self, X_test): 
        res = []
        for c in classes:
            p_y = class_prior[c] 
            p_x_y = 1
            for i in X_test.items():
                p_x_y *= prior[tuple(list(i)+[c])] 
            res.append(p_y*p_x_y)
                
        return classes[np.argmax(res)]

if __name__=="__main__":
    X1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    X2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    df = pd.DataFrame({'x1':X1, 'x2':X2, 'y':y})
    X = df[['x1', 'x2']]
    y = df[['y']]
    X_test = {'x1': 2, 'x2': 'S'}
    nb = naive_bayes()
    classes, class_prior, prior = nb.fit(X, y)
    print('prediction for test data:', nb.predict(X_test))