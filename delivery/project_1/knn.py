# Code for performing and visualizing knn

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def knn(X_train,y_train):


        clf = KNeighborsClassifier(metric="minkowski")
        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {"n_neighbors": np.arange(2, 10)}
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(clf, param_grid, cv=5)
        #fit model to data
        knn_gscv.fit(X_train, y_train)
        #check top performing n_neighbors value
        k = knn_gscv.best_params_['n_neighbors']
        print("best k-param selected:",k)

        #Create the model
        neigh = KNeighborsClassifier(n_neighbors=k, metric="minkowski")

        # cross validation score
        # train model and calculate cross validation score 
        # number of folds: 10
        # score is the accuracy
        scores = cross_val_score(neigh, X_train, y_train, cv = 10)
        mean = np.mean(scores)
        print(f"mean score of training data with {k}-nn :", mean)


        """


        """
        return k,neigh,mean