from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def adaboost(X_train, X_test, y_train):
    model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_depth=8) ,n_estimators=100)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels


def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier(criterion='entropy', n_estimators=200, max_features='sqrt')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels