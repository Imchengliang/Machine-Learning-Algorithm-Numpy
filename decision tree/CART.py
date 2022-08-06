import numpy as np 
import pandas as pd 
from math import log

class CartTree:
    # define a node class
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

    def gini(self, nums):
        probs = [nums.count(i)/len(nums) for i in set(nums)]
        gini = sum([p*(1-p) for p in probs])
        return gini

    def split_dataframe(self, data, col): 
        # unique value of column
        unique_values = data[col].unique()
        # empty dict of dataframe
        result_dict = {elem : pd.DataFrame for elem in unique_values}
        # split dataframe based on column value
        for key in result_dict.keys():
            result_dict[key] = data[:][data[col] == key]

        return result_dict

    def choose_best_col(self, df, label): 
        # Calculating label's gini index
        gini_D = self.gini(df[label].tolist())
        # columns list except label
        cols = [col for col in df.columns if col not in [label]]
        # initialize the max infomation gain, best column and best splited dict
        min_value, best_col, min_splited = 999, None, None
        # split data based on different column 
        for col in cols:
            splited_set = self.split_dataframe(df, col) 
            gini_DA = 0
            for subset_col, subset in splited_set.items():
                # calculating splited dataframe label's gini index
                gini_Di = self.gini(subset[label].tolist())
                # calculating gini index of current feature
                gini_DA += len(subset)/len(df) * gini_Di
                if gini_DA < min_value:
                    min_value, best_col = gini_DA, col
                    min_splited = splited_set
                    
        return min_value, best_col, min_splited

    # print the tree
    def print_tree(self, node, tabs):
        print(tabs + node.name)
        for connection, child_node in node.connections.items():
            print(tabs + "\t" + "(" + str(connection) + ")") 
            self.print_tree(child_node, tabs + "\t\t")

    def construct_tree(self):
        self.construct(self.root, "", self.data, self.columns)

    # construct tree
    def construct(self, parent_node, parent_connection_label, input_data, columns): 
        min_value, best_col, min_splited = self.choose_best_col(input_data[columns], self.label)
        if not best_col:
            node = self.Node(input_data[self.label].iloc[0]) 
            parent_node.connect(parent_connection_label, node) 
            return

        node = self.Node(best_col) 
        parent_node.connect(parent_connection_label, node)

        new_columns = [col for col in columns if col != best_col] 
        # Recursively constructing decision trees
        for splited_value, splited_data in min_splited.items():
            self.construct(node, splited_value, splited_data, new_columns)

if __name__ == '__main__':
    df = pd.read_csv('/Users/imchengliang/Downloads/Code/ML/decision tree/example_data.csv', dtype={'windy': 'str'})
    tree = CartTree(df, 'play')
    tree.construct_tree()
    tree.print_tree(tree.root, "")