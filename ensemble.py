from sklearn.ensemble import RandomForestClassifier

def rfc(X_train, X_test, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels