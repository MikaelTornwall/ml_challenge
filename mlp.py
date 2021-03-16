from sklearn.neural_network import MLPClassifier


def mlp(X_train, X_test, y_train):
    model = MLPClassifier(max_iter=1000)
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels