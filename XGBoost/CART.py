import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# define binary feature split function
def feature_split(X, feature_i, threshold):
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_left = np.array([sample for sample in X if split_func(sample)])
    X_right = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_left, X_right])

def calculate_gini(y):
    y = y.tolist()
    probs = [y.count(i)/len(y) for i in np.unique(y)]
    gini = sum([p*(1-p) for p in probs])
    return gini

def data_shuffle(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def cat_label_convert(y, n_col=None):
    if not n_col:
        n_col = np.amax(y) + 1
    one_hot = np.zeros((y.shape[0], n_col))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

# define the node
class TreeNode():
    def __init__(self, feature_i=None, threshold=None, leaf_value=None, left_branch=None, right_branch=None): 
        # feature index
        self.feature_i = feature_i
        # feature partition threshold
        self.threshold = threshold
        # value of leaf
        self.leaf_value = leaf_value
        # left subtree
        self.left_branch = left_branch     
        # right subtree
        self.right_branch = right_branch

# define a decision tree
class BinaryDecisionTree(object):
    def __init__(self, min_samples_split=2, min_gini_impurity=999, max_depth=float("inf"), loss=None):
        # root node
        self.root = None
        # minimum number of split samples of a node
        self.min_samples_split = min_samples_split
        # initialized gini impurity of a node
        self.mini_gini_impurity = min_gini_impurity
        # max depth of a tree
        self.max_depth = max_depth
        # computation of gini impurity
        self.gini_impurity_calculation = None
        # prediction function for leaf node value
        self._leaf_value_calculation = None
        # loss function
        self.loss = loss

    def fit(self, X, y, loss=None):
        # build the decision tree recursively
        self.root = self._build_tree(X, y)
        self.loss=None

    # function for constructing a decision tree
    def _build_tree(self, X, y, current_depth=0):
        # initialize gini impurity
        init_gini_impurity = 999
        # initialize feature partition threshold
        best_criteria = None 
        # initialize data samples   
        best_sets = None 

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # merge input and label
        Xy = np.concatenate((X, y), axis=1)
        # obtain number of sample and feature
        n_samples, n_features = X.shape
        # set up the condition for building a decision tree
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                # obtain all values of i-th feature
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                # obtain the unique value of i-th feature
                unique_values = np.unique(feature_values)

                # traverse the values and find the best feature splitting threshold
                for threshold in unique_values:
                    # binary splitting of feature nodes
                    Xy1, Xy2 = feature_split(Xy, feature_i, threshold)
                    # size of both subsets need to be larger than 0
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # obtain the label values of subsets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # calculate gini impurity
                        # impurity = self.impurity_calculation(y, y1, y2)
                        impurity = calculate_gini(y)

                        # update minimum gini impurity, feature index and
                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   
                                "lefty": Xy1[:, n_features:],   
                                "rightX": Xy2[:, :n_features],  
                                "righty": Xy2[:, n_features:]   
                                }

        # build subtree if the gini impurity is smaller the the initialized value
        if init_gini_impurity < self.mini_gini_impurity:
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return TreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"], left_branch=left_branch, right_branch=right_branch)

        #calculate leaf value
        leaf_value = self._leaf_value_calculation(y)
        return TreeNode(leaf_value=leaf_value)

    # define prediction function for binary tree
    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
    
        if tree.leaf_value is not None:
            return tree.leaf_value
        # obtain the selected feature value
        feature_value = x[tree.feature_i]
        # determine left or right subtree
        branch = tree.right_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.right_branch

        return self.predict_value(x, branch)

    # prediction function for data set
    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

class ClassificationTree(BinaryDecisionTree):
    def _calculate_gini_impurity(self, y, y1, y2):
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_impurity = p * calculate_gini(y1) + (1-p) * calculate_gini(y2)
        return gini_impurity
    
    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # count the majority vote
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
    
    def fit(self, X, y):
        self.impurity_calculation = self._calculate_gini_impurity
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

class RegressionTree(BinaryDecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = np.var(y, axis=0)
        var_y1 = np.var(y1, axis=0)
        var_y2 = np.var(y2, axis=0)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_y1 + frac_2 * var_y2)
        return sum(variance_reduction)

    # get the mean of node value
    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self.impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

if __name__ == '__main__':
    # test classification tree
    from sklearn import datasets
    data = datasets.load_iris()
    X, y = data.data, data.target
    y = y.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X, y.reshape(-1,1), test_size=0.3)
    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    # test regression tree
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    #from sklearn.datasets import fetch_california_housing
    #X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RegressionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    from sklearn.tree import DecisionTreeRegressor
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

