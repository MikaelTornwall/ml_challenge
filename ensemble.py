from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def adaboost(X_train, X_test, y_train):
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8), n_estimators=100)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def stacking(X_train, X_test, y_train, y_test):        
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'))]    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    score = model.fit(X_train, y_train).score(X_test, y_test)
    return score


# ---------------------- for final submission ---------------------- 


def final_classifier(X_train, X_test, y_train):    
    classifiers = [('qda', QuadraticDiscriminantAnalysis()), ('lda', LinearDiscriminantAnalysis()), ('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'))]    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


# -------------------------------------------------------------------