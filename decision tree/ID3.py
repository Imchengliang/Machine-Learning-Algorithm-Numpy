import numpy as np 
import pandas as pd 
from math import log

def entropy(ele): 
    # Calculating the probability distribution of list value 
    probs = [ele.count(i)/len(ele) for i in set(ele)]
    # Calculating entropy value
    entropy = -sum([prob*log(prob, 2) for prob in probs]) 
    return entropy

def split_dataframe(data, col): 
    # unique value of column
    unique_values = data[col].unique()
    # empty dict of dataframe
    result_dict = {elem : pd.DataFrame for elem in unique_values} 
    # split dataframe based on column value
    for key in result_dict.keys():
        result_dict[key] = data[:][data[col] == key] 
        
    return result_dict

def choose_best_col(df, label): 
    # Calculating label's entropy
    entropy_D = entropy(df[label].tolist())
    # columns list except label
    cols = [col for col in df.columns if col not in [label]]
    # initialize the max infomation gain, best column and best splited dict 
    max_value, best_col = -999, None
    max_splited = None
    # split data based on different column
    for col in cols:
        splited_set = split_dataframe(df, col) 
        entropy_DA = 0
        for subset_col, subset in splited_set.items():
            # calculating splited dataframe label's entropy
            entropy_Di = entropy(subset[label].tolist())
            # calculating entropy of current feature
            entropy_DA += len(subset)/len(df) * entropy_Di
        # calculating infomation gain of current feature
        info_gain = entropy_D - entropy_DA 
        if info_gain > max_value:
            max_value, best_col = info_gain, col
            max_splited = splited_set

    return max_value, best_col, max_splited

class ID3Tree:
    # define a Node class class Node:
    class Node:
        def __init__(self, name): 
            self.name = name
            self.connections = {}

        def connect(self, label, node): 
            self.connections[label] = node

    def __init__(self, data, label): 
        self.columns = data.columns 
        self.data = data
        self.label = label
        self.root = self.Node("Root")

    # print tree method
    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + str(connection) + ")")
            self.print_tree(child_node, tabs + "\t\t") 
            
    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

    # construct tree
    def construct(self, parent_node, parent_connection_label, input_data, columns): 
        max_value, best_col, max_splited = choose_best_col(input_data[columns], self.label)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0])
            parent_node.connect(parent_connection_label, node)
            return

        node = self.Node(best_col) 
        parent_node.connect(parent_connection_label, node)
        new_columns = [col for col in columns if col != best_col] 
        # Recursively constructing decision trees
        for splited_value, splited_data in max_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)

if __name__ == '__main__':
    df = pd.read_csv('example_data.csv')
    tree = ID3Tree(df, 'play')
    tree.construct_tree()
    tree.print_tree(tree.root, "")

# The following part uses sklearn to implement a decision tree    
'''
from sklearn.datasets import load_iris from sklearn import tree
import graphviz
iris = load_iris()
# choose entropy in criterion is choosing ID3
clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best') 
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
'''