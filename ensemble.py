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
    # model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def stacking(X_train, X_test, y_train, y_test):
    # Best so far
    # classifiers = [('rf', RandomForestClassifier()), ('ab', AdaBoostClassifier(base_estimator=GaussianNB())), ('qda', QuadraticDiscriminantAnalysis())]
    classifiers = [('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt')), ('lda', LinearDiscriminantAnalysis()), ('qda', QuadraticDiscriminantAnalysis())] # best performer
    # model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=5)
    # score = model.fit(X_train, y_train).score(X_test, y_test)
    
    # classifiers = [('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt')), ('lda', LinearDiscriminantAnalysis()), ('qda', QuadraticDiscriminantAnalysis())]
    # classifiers = [('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier(10)), ('ab', AdaBoostClassifier(base_estimator=GaussianNB())), ('lda', LinearDiscriminantAnalysis()), ('qda', QuadraticDiscriminantAnalysis())]
    # classifiers = [('rf', RandomForestClassifier()), ('ab', AdaBoostClassifier(base_estimator=GaussianNB())), ('qda', QuadraticDiscriminantAnalysis())]
    # classifiers = [('rf', RandomForestClassifier()), ('ab', AdaBoostClassifier()), ('qda', QuadraticDiscriminantAnalysis())]
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=10)
    score = model.fit(X_train, y_train).score(X_test, y_test)
    return score

# for final submission
def final_classifier(X_train, X_test, y_train):
    classifiers = [('rf', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt')), ('lda', LinearDiscriminantAnalysis()), ('qda', QuadraticDiscriminantAnalysis())] # best performer    
    model = StackingClassifier(classifiers, final_estimator=LinearDiscriminantAnalysis(), cv=5)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels