# Code for performing and visualizing LR

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def lr(X_train,y_train):


    # create model
    clf = LogisticRegression()

    # cross validation score
    # train model and calculate cross validation score 
    # number of folds: 10
    # score is the accuracy
    scores = cross_val_score(clf, X_train, y_train, cv = 10)
    mean = np.mean(scores)
    print(f"mean score of training data with LR :", mean)

    return clf, mean