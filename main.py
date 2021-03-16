import numpy as np
import pandas as pd
import random, math
from sklearn.model_selection import KFold
import data, knn, da, mlp, svm, ensemble


#
# Models to try (out-of-the-box results with scikit learn)
#
# K-nearest neighbor                                    - ~0.6 with k = 20
# Naive Bayes
# Linear Discriminative Analysis                        - ~0.55 (clearly not linear relationship)
# ---------------------------------- High performer -------------------------------------------------
# Quadratic Discriminative Analysis                     - +0.70
# ---------------------------------------------------------------------------------------------------
# Logistic Regression (penalty l1)
# Ensemble learning
#   Adaboost                                            - ~0.40
#   Bagging
# ---------------------------------- High performer -------------------------------------------------
#   Random Forests (suited for multi-class problems)    - +0.70
# ---------------------------------------------------------------------------------------------------
# SVM                                                   - 0.684, kernel='poly'
# ---------------------------------- High performer -------------------------------------------------
# Multilayer Perceptron                                 - +0.70, max_iter=1000
# ---------------------------------------------------------------------------------------------------
#


# 
# To do
# [x] Shuffle data
# [x] K-fold crossvalidation 
# 
# [ ] PCA
#
# [ ] Try manually deleting feature vectors
# [ ] Check x8 for outliers
#


def ten_fold_CV(X, y):
    cv = KFold(n_splits=10, shuffle=True)    
    return cv.split(X, y)


def train(X, y, splits, model, title):
    accuracies = []

    for train_i, test_i in splits:
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]

        labels = model(X_train, X_test, y_train)
        correctly_labeled = np.array(np.where(y_test == labels))
        accuracy = correctly_labeled[0].shape[0] / y_test.shape[0]
        accuracies.append(accuracy)

    accuracies = np.array(accuracies)
    print(f'{title}: {np.mean(accuracies)}')


def main():
    df = data.getData('TrainOnMe.csv')    
    X, y = data.cleanData(df)
    # print(X.shape)
    # print(y.shape)
    
    splits = ten_fold_CV(X, y)

    # models = [knn.knn, da.lda, da.qda, mlp.mlp, svm.svm, ensemble.rfc, ensemble.adaboost]
    # model_titles = ['20-NN', 'LDA', 'QDA', 'Multi-layer Perceptron', 'SVM', 'Random Forest', 'Adaboost']

    models = [knn.knn, da.lda, da.qda, svm.svm, ensemble.rfc, ensemble.adaboost]
    model_titles = ['20-NN', 'LDA', 'QDA', 'SVM', 'Random Forest', 'Adaboost']

    for i in range(len(models)):
        splits = ten_fold_CV(X, y)
        train(X, y, splits, models[i], model_titles[i])

    # split = int(round(X.shape[0] * 0.8))
    
    # X_train = X[:split]
    # X_test = X[split:]
    # y_train = y[:split]
    # y_test = y[split:]

    # qda_labels = da.qda(X_train, X_test, y_train)
    # qda_correctly_labeled = np.array(np.where(y_test == qda_labels))
    # print("QDA:", qda_correctly_labeled[0].shape[0] / y_test.shape[0])    


if __name__ == "__main__":
    main()