# import python ML modules
import numpy as np
import matplotlib.pyplot as plt

# 3 methods for classification task: KNN (local rule), QDA (global rule), logistic regression (global rule).
# Assume binary classification task with p=2 (2 features x_1 and x_2).

# 4 Assumptions for dataset distributions
# 1. Small dataset with less overlap (1)
# 2. Small dataset with large overlap (2) (simple boundary)
# 3. Large dataset with less overlap (3)
# 4. Large dataset with large overlap (4) (simple boundary)

# keys: data distribution (1, 2, 3 or 4)
# values: n_samples, means, covariance matrix
meta_data = {
    1: [30, [[0,0], [1,1]], [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]],
    2: [30, [[0,0], [2,2]], [[[1, 0], [0, 1]], [[1, 0.3], [0.3, 1]]]],
    3: [1000, [[0,0], [1,1]], [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]],
    4: [1000, [[0,0], [2,2]], [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]],
}

# Logistic regression: [1000, [[0,0], [2,2]], [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]]
# QDA: [1000, [[0,0], [2,2]], [[[1, 0.5], [0.5, 1]], [[1, 0], [0, 1.5]]]]
# KNN: [1000, [[0,0], [2,2]], [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]]


# select key and values
# TODO: Change key here (1,2,3,4)
key = 4
data_distrib_info = meta_data[key]

# Set the number of samples per class (Assumption: balanced data sets)
n_samples = data_distrib_info[0]

# Define the mean and covariance for each class
means = np.array(data_distrib_info[1])
# Intuitive understanding:
# Main diagonal: variances of x_1 and x_2 (scaling of "original" variances 1)
# Entries on the side: covariances of x_1 and x_2 (transform them actually into
# "changed shape")
# First entry: class 0, second entry: class 1
covs = np.array(data_distrib_info[2])

# Generate the data for each class
# class 0
X0 = np.random.multivariate_normal(means[0], covs[0], n_samples)
y0 = np.zeros(n_samples)
# class 1
X1 = np.random.multivariate_normal(means[1], covs[1], n_samples)
y1 = np.ones(n_samples)

# Combine the data from both classes
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((y0, y1), axis=0)

np.save(f"X_{key}", X)
np.save(f"y_{key}", y)

# Plot the data
plt.scatter(X[:,0], X[:,1], c=y, alpha=0.4)
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig(f"data_{key}")
