import numpy as np

def cat_label_convert(y, n_col=None):
    if not n_col:
        n_col = np.amax(y) + 1
    one_hot = np.zeros((y.shape[0], n_col))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot