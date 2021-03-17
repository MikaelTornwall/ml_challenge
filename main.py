import numpy as np
import pandas as pd
import random, math
from sklearn.model_selection import KFold
import data, knn, da, mlp, svm, ensemble
from sklearn.decomposition import PCA


def write_file(labels):
    f = open("labels.txt", "w")        
    for label in labels:
        f.write(f'{label}\n')
    f.close()


def ten_fold_CV(X, y):
    cv = KFold(n_splits=10, shuffle=True)    
    return cv.split(X, y)


def train(X, y, splits, model, title):
    accuracies = []
    stacked = []

    for train_i, test_i in splits:
        X_train, X_test = X[train_i], X[test_i]
        y_train, y_test = y[train_i], y[test_i]

        labels = model(X_train, X_test, y_train)
        correctly_labeled = np.array(np.where(y_test == labels))
        accuracy = correctly_labeled[0].shape[0] / y_test.shape[0]
        accuracies.append(accuracy)

        score = ensemble.stacking(X_train, X_test, y_train, y_test)
        stacked.append(score)

    accuracies = np.array(accuracies)
    print(f'{title}: {np.mean(accuracies)}')
    stacked = np.array(stacked)
    print(f'stacking: {np.mean(stacked)}')


def main():
    df = data.getData('TrainOnMe.csv')    
    X, y = data.cleanData(df)
    
    splits = ten_fold_CV(X, y)
    models = [knn.knn, da.lda, da.qda, mlp.mlp, svm.svm, ensemble.rfc, ensemble.adaboost]
    model_titles = ['20-NN', 'LDA', 'QDA', 'Multi-layer Perceptron', 'SVM', 'Random Forest', 'Adaboost']

    # two features explain very little of the variance, namely x2 and x3
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    print(pca.explained_variance_ratio_)

    # for i in range(len(models)):
    #     splits = ten_fold_CV(X, y)
    #     train(X, y, splits, models[i], model_titles[i])

    # ---------------- For final submission -------------------
    #
    # 
    X_eval = data.get_evaluation_data()    
    final_labels = ensemble.final_classifier(X, X_eval, y)
    write_file(final_labels)
    #
    #
    # ---------------------------------------------------------


if __name__ == "__main__":
    main()