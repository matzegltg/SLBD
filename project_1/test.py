import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from knn import knn

X = np.load(f"X_{4}.npy")
y = np.load(f"y_{4}.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42,
                                                            shuffle=True)
k,model_knn, mean_knn = knn(X_train,y_train)
model_knn.fit(X_train,y_train)
#Plotting the result
# create "test" meshgrid
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

# predict class using "test" meshgrid
y = model_knn.predict(np.c_[xx1.ravel(), xx2.ravel()])
y = y.reshape(xx1.shape)

# plot results of "test" meshgrid
fig1 = plt.figure("Figure 1")
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.contour(xx1, xx2, y, [0.0], linewidths=1)
plt.show()