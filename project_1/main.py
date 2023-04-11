import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from knn import knn
from LR import lr 
from QDA import qda 

# If true: Evaluate models with simulated data, else: real world data
real_data = True

#load data
for key in [1,2,3,4]:
    
    # load generated data
    if real_data:
        X = np.load(f"X_rw_{key}.npy")
        y = np.load(f"y_rw_{key}.npy")
    else:
        X = np.load(f"X_{key}.npy")
        y = np.load(f"y_{key}.npy")

    # split dataset into 80% training/validation data and 20% unseen test data
    # select random state and shuffle data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42,
                                                            shuffle=True)
    results = []
    
    k,model_knn, mean_knn = knn(X_train,y_train)
    model_qda, mean_qda = qda(X_train,y_train)
    model_lr, mean_lr = lr(X_train,y_train)

    #Case of a tie between two models
    if (mean_knn == max(mean_knn, mean_qda,mean_lr) or mean_qda == max(mean_knn, mean_qda,mean_lr))and abs(mean_knn-mean_qda) < 0.001:
        print("Tie (KNN/QDA)--- Cross validation with 5 folds")
        scores = cross_val_score(model_knn, X_train, y_train, cv = 5)
        mean_knn = np.mean(scores)
        scores = cross_val_score(model_qda, X_train, y_train, cv = 5)
        mean_qda = np.mean(scores)

    if (mean_knn == max(mean_knn, mean_qda,mean_lr) or mean_lr == max(mean_knn, mean_qda,mean_lr)) and abs(mean_knn-mean_lr) < 0.001:
        print("Tie (KNN/LR)--- Cross validation with 5 folds")
        scores = cross_val_score(model_knn, X_train, y_train, cv = 5)
        mean_knn = np.mean(scores)
        scores = cross_val_score(model_lr, X_train, y_train, cv = 5)
        mean_lr = np.mean(scores)

    if( mean_qda == max(mean_knn, mean_qda,mean_lr) or mean_lr == max(mean_knn, mean_qda,mean_lr)) and abs(mean_qda-mean_lr) < 0.001:
        print("Tie (LR/QDA)--- Cross validation with 5 folds")
        scores = cross_val_score(model_qda, X_train, y_train, cv = 5)
        mean_qda = np.mean(scores)
        scores = cross_val_score(model_lr, X_train, y_train, cv = 5)
        mean_lr = np.mean(scores)

    #fit models
    model_knn.fit(X_train,y_train)
    model_qda.fit(X_train,y_train)
    model_lr.fit(X_train,y_train)
    
    #predict
    y_pred_knn = model_knn.predict(X_test)
    y_pred_qda = model_qda.predict(X_test)
    y_pred_lr = model_lr.predict(X_test)

    # calculate confusion matrix for each model
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_qda = confusion_matrix(y_test, y_pred_qda)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    # calculate accuracy
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_qda = accuracy_score(y_test, y_pred_qda)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # if knn has the best accuracy
    if mean_knn == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is {k}-NN")
        model = model_knn
        acc = acc_knn
        method = str(k)+"-NN"

    # if qda has the best accuracy
    elif mean_qda == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is QDA")
        model = model_qda
        acc = acc_qda
        method = "QDA"
    # if lr has the best accuracy
    elif mean_lr == max(mean_knn, mean_qda,mean_lr):
        print(f"Best prediction model for dataset{key} is Logistic Regression")
        model = model_knn
        acc = acc_lr
        method = "Logistic  Regression"

    if real_data:
        pass
    else:
        #Plotting the result
        # create "test" meshgrid
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, .1), np.arange(x2_min, x2_max, .1))

        # predict class using "test" meshgrid
        y = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
        y = y.reshape(xx1.shape)

        # plot results of "test" meshgrid
        fig1 = plt.figure("Figure 1")
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        plt.pcolormesh(xx1, xx2, y, cmap=ListedColormap(['#FFA4FF', '#C0C9FF']))

        # scatter training data with corresponding classification
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=ListedColormap(['#FF6A4C', '#526AFF']), alpha=0.4, marker="+", label="training")

        # scatter test data with corresponding classification
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=ListedColormap(['#AD001F', '#010086']), alpha=0.4, marker=".", label="testing")

        # visualization of the legend
        legend_handles = [Patch(color='#FF6A4C', label='Trainset class 0'), Patch(color='#AD001F', label='Testset class 0'), Patch(color='#526AFF', label='Trainset class 1'),   Patch(color='#010086', label='Testset class 1')]
        plt.legend(handles=legend_handles, ncol=4, bbox_to_anchor=[0.5, 0], loc='lower center', fontsize=8, handlelength=.8)

        plt.title(f"{method}\n Tested accuracy: {np.round(acc, 2)}")
        # store image
        plt.savefig(f"figures/{method}_dataset{key}")
            

        

    # calculate precision
    precision_knn = precision_score(y_test, y_pred_knn)
    precision_qda = precision_score(y_test, y_pred_qda)
    precision_lr = precision_score(y_test, y_pred_lr)

    # calculate recall
    recall_knn = recall_score(y_test, y_pred_knn)
    recall_qda = recall_score(y_test, y_pred_qda)
    recall_lr = recall_score(y_test, y_pred_lr)

    # calculate F1-score
    f1_knn = f1_score(y_test, y_pred_knn)
    f1_qda = f1_score(y_test, y_pred_qda)
    f1_lr = f1_score(y_test, y_pred_lr)

    # print the results
    print("Accuracy: KNN:", acc_knn, "QDA: ", acc_qda, "LR: ", acc_lr)
    print("Precision: KNN:", precision_knn, "QDA: ", precision_qda, "LR: ", precision_lr)
    print("Recall: KNN:", recall_knn, "QDA: ", recall_qda, "LR: ", recall_lr)
    print("F1-score: KNN:", f1_knn, "QDA: ", f1_qda, "LR: ", f1_lr)
    print("\n")

        
    # plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    sns.heatmap(cm_knn, annot=True, fmt='g', ax=axes[0])
    sns.heatmap(cm_qda, annot=True, fmt='g', ax=axes[1])
    sns.heatmap(cm_lr, annot=True, fmt='g', ax=axes[2])
    
    axes[0].set_title('KNN')
    axes[1].set_title('QDA')
    axes[2].set_title('Logistic Regression')
    
    plt.tight_layout()
    if real_data:
        plt.savefig(f"figures/confusion_mat/matrix_rw_{key}")
    else:
        plt.savefig(f"figures/confusion_mat/matrix{key}")
    fig.canvas.flush_events()
        