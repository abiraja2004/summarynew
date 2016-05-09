__author__ = 'HyNguyen'

import numpy as np
import sys
import copy
from tree import *

def load_sentiment_data():
    with open("../data/rt-polarity.neg.out.txt", mode="r") as f:
        neg_trees_str = f.readlines()

    X_neg = []
    for neg_tree_str in neg_trees_str:
        t = Tree(neg_tree_str)
        t = merge_bin_tree(t)
        t.label = 0
        X_neg.append(t)
    print(len(X_neg))

    with open("../data/rt-polarity.pos.out.txt", mode="r") as f:
        pos_trees_str = f.readlines()

    X_pos = []
    for pos_tree_str in pos_trees_str:
        t = Tree(pos_tree_str)
        t = merge_bin_tree(t)
        t.label = 1
        X_pos.append(t)
    print(len(X_pos))

    X = X_neg + X_pos
    np.random.shuffle(X)

    trainnum = int(0.8*len(X))
    X_train = X[:trainnum]
    X_valid = X[trainnum:]
    sys.stdout.write("{0}, {1} \n".format(len(X_train), len(X_valid)))
    sys.stdout.flush()

    return X_train , X_valid

if __name__ == "__main__":

    X_train , X_valid = load_sentiment_data()
    print("ttdtilu hhididly")
