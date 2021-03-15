import numpy as np
import pandas as pd
import random, math
from sklearn.model_selection import KFold
import data, knn, qda, mlp, svm, ensemble


#
# Models to try (out-of-the-box results with scikit learn)
#
# K-nearest neighbor                                    - 0.617 with k = 20
# Naive Bayes
# Linear Discriminative Analysis                        - 0.570 (clearly not linear relationship)
# Quadratic Discriminative Analysis                     - 0.721
# Logistic Regression (penalty l1)
# Ensemble learning
#   Boosting
#   Bagging
#   Random Forests (suited for multi-class problems)    - 0.738
# SVM                                                   - 0.594, kernel='poly'
# Multilayer Perceptron                                 - 0.688, max_iter=350
#

# 
# To do
# [ ] Shuffle data
# [ ] K-fold crossvalidation 
# 
# [ ] PCA
#

def ten_fold_CV(X):
    cv = KFold(n_splits=10)
    return cv.get_n_splits(X)

def main():
    df = data.getData('TrainOnMe.csv')    
    X, y = data.cleanData(df)
    print(X.shape)
    print(y.shape)

    split = int(round(X.shape[0] * 0.7))
    
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    knn_labels = knn.kNN(20, X_train, X_test, y_train)
    knn_correctly_labeled = np.array(np.where(y_test == knn_labels))
    print("KNN:", knn_correctly_labeled[0].shape[0] / y_test.shape[0])
    
    lda_labels = qda.lda(X_train, X_test, y_train)
    lda_correctly_labeled = np.array(np.where(y_test == lda_labels))
    print("LDA:", lda_correctly_labeled[0].shape[0] / y_test.shape[0])

    qda_labels = qda.qda(X_train, X_test, y_train)
    qda_correctly_labeled = np.array(np.where(y_test == qda_labels))
    print("QDA:", qda_correctly_labeled[0].shape[0] / y_test.shape[0])    

    mlp_labels = mlp.mlp(X_train, X_test, y_train)
    mlp_correctly_labeled = np.array(np.where(y_test == mlp_labels))
    print("MLP:", mlp_correctly_labeled[0].shape[0] / y_test.shape[0])    

    svm_labels = svm.svm(X_train, X_test, y_train)
    svm_correctly_labeled = np.array(np.where(y_test == svm_labels))
    print("SVM:", svm_correctly_labeled[0].shape[0] / y_test.shape[0])

    rfc_labels = ensemble.rfc(X_train, X_test, y_train)
    rfc_correctly_labeled = np.array(np.where(y_test == rfc_labels))
    print("Random Forest:", rfc_correctly_labeled[0].shape[0] / y_test.shape[0])    
    

if __name__ == "__main__":
    main()