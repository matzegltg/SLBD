import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score
from knn import knn
from LR import lr 
from QDA import qda 


#load data
for key in [1,2,3,4]:
    # load generated data
    X = np.load(f"X_{key}.npy")
    y = np.load(f"y_{key}.npy")

    # split dataset into 80% training/validation data and 20% unseen test data
    # select random state and shuffle data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42,
                                                            shuffle=True)
    results = []
    
    k,model_knn, mean_knn = knn(X,y)
    model_qda, mean_qda = qda(X,y)
    model_lr, mean_lr = lr(X,y)

    # if knn has the best accuracy
    if mean_knn == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is {k}-NN")

        #fit the model
        model_knn.fit(X_train,y_train)

        #predictions on training set
        acc = accuracy_score(y_test, model_knn.predict(X_test))

        # create "test" meshgrid
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

        # predict class using "test" meshgrid
        y = model_knn.predict(np.c_[xx1.ravel(), xx2.ravel()])
        y = y.reshape(xx1.shape)

        # plot results of "test" meshgrid
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.pcolormesh(xx1, xx2, y, cmap=ListedColormap(['#FFA4FF', '#C0C9FF']))

        # scatter training data with corresponding classification
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['#FF6A4C', '#526AFF']), alpha=0.4, marker="+", label="training")

        # scatter test data with corresponding classification
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=ListedColormap(['#AD001F', '#010086']), alpha=0.4, marker=".", label="testing")

        # visualization of the legend
        legend_handles = [Patch(color='#FF6A4C', label='Trainset class 0'),  Patch(color='#526AFF', label='Trainset class 1'), Patch(color='#AD001F', label='Testset class 0'),  Patch(color='#010086', label='Testset class 1')]
        plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 0], loc='lower center', fontsize=8, handlelength=.8)

        plt.title(f"{k}nn-{key}\n Tested accuracy: {np.round(acc, 2)}")
        # store image
        plt.savefig(f"figures/knn/{k}nn-{key}new")
        

    # if qda has the best accuracy
    if mean_qda == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is QDA")

        #fit the model
        model_qda.fit(X_train,y_train)

        #predictions
        acc = accuracy_score(y_test, model_qda.predict(X_test))

        # create "test" meshgrid
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

        # predict class using "test" meshgrid
        y = model_qda.predict(np.c_[xx1.ravel(), xx2.ravel()])
        y = y.reshape(xx1.shape)

        # plot results of "test" meshgrid
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.pcolormesh(xx1, xx2, y, cmap=ListedColormap(['#FFA4FF', '#C0C9FF']))

        # scatter training data with corresponding classification
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['#FF6A4C', '#526AFF']), alpha=0.4, marker="+", label="training")

        # scatter test data with corresponding classification
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=ListedColormap(['#AD001F', '#010086']), alpha=0.4, marker=".", label="testing")

        # visualization of the legend
        legend_handles = [Patch(color='#FF6A4C', label='Trainset class 0'),  Patch(color='#526AFF', label='Trainset class 1'), Patch(color='#AD001F', label='Testset class 0'),  Patch(color='#010086', label='Testset class 1')]
        plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 0], loc='lower center', fontsize=8, handlelength=.8)

        plt.title(f"QDA \nTested accuracy: {np.round(acc, 2)}")
        # store image
        plt.savefig(f"figures/qda/QDAnew")
        
    
    # if lr has the best accuracy
    if mean_lr == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is Logistic Regression")

        #fit the model
        model_lr.fit(X_train,y_train)

        #predictions
        acc = accuracy_score(y_test, model_lr.predict(X_test))

        # create "test" meshgrid
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

        # predict class using "test" meshgrid
        y = model_lr.predict(np.c_[xx1.ravel(), xx2.ravel()])
        y = y.reshape(xx1.shape)

        # plot results of "test" meshgrid
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.pcolormesh(xx1, xx2, y, cmap=ListedColormap(['#FFA4FF', '#C0C9FF']))

        # scatter training data with corresponding classification
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['#FF6A4C', '#526AFF']), alpha=0.4, marker="+", label="training")

        # scatter test data with corresponding classification
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=ListedColormap(['#AD001F', '#010086']), alpha=0.4, marker=".", label="testing")

        # visualization of the legend
        legend_handles = [Patch(color='#FF6A4C', label='Trainset class 0'),  Patch(color='#526AFF', label='Trainset class 1'), Patch(color='#AD001F', label='Testset class 0'),  Patch(color='#010086', label='Testset class 1')]
        plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 0], loc='lower center', fontsize=8, handlelength=.8)

        plt.title(f"Logitic Regression \n Tested accuracy: {np.round(acc, 2)}")
        # store image
        plt.savefig(f"figures/lr/LR_{key}new")
        