# Code for performing and visualizing knn

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# load generated data
X = np.load("X.npy")
y = np.load("y.npy")

# split dataset into 80% training/validation data and 20% unseen test data
# select random state and shuffle data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42,
                                                    shuffle=True)

# create model
# two choieces for knn: metric, number of neighbours k
# basic assumptions: k=3, metrix minkowsky -> euclidian metric for p = 2

# initialize the model
n_neighbours=3
neigh = KNeighborsClassifier(n_neighbors=n_neighbours, metric="minkowski")

# cross validation score
# train model and calculate cross validation score 
# number of folds: 10
# score is the accuracy
scores = cross_val_score(neigh, X_train, y_train, cv = 10)
print(scores)

# train model
neigh.fit(X_train, y_train)

# create "test" meshgrid
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

# predict class using "test" meshgrid
y = neigh.predict(np.c_[xx1.ravel(), xx2.ravel()])
y = y.reshape(xx1.shape)

# plot results of "test" meshgrid
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.pcolormesh(xx1, xx2, y, cmap=ListedColormap(['#FFA4FF', '#C0C9FF']))

# scatter training data with corresponding classification
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['#FF6A4C', '#526AFF']), alpha=0.8, marker="+", label="training")

# scatter test data with corresponding classification
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=ListedColormap(['#AD001F', '#010086']), alpha=0.9, marker=".", label="testing")

# visualization of the legend
legend_handles = [Patch(color='#FF6A4C', label='Trainset class 0'),  Patch(color='#526AFF', label='Trainset class 1'), Patch(color='#AD001F', label='Testset class 0'),  Patch(color='#010086', label='Testset class 1')]
plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)

# store image
plt.savefig(f"knn_{n_neighbours}")