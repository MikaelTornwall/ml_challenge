import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def qda(X_train, X_test, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels

def lda(X_train, X_test, y_train):
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels
