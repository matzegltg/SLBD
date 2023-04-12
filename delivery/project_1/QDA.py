# Code for performing and visualizing QDA

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

def qda(X_train,y_train):

    # create model
    clf = QuadraticDiscriminantAnalysis()

    # cross validation score
    # train model and calculate cross validation score 
    # number of folds: 10
    # score is the accuracy
    scores = cross_val_score(clf, X_train, y_train, cv = 10)
    mean = np.mean(scores)
    print(f"mean score of training data with QDA :", mean)

    """

    """
    return clf, mean