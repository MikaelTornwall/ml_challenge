import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# KNN of Unknown Data Point
# To classify the unknown data point using the KNN (K-Nearest Neighbor) algorithm:
#     Normalize the numeric data (check if necessary with sklearn)
#     Find the distance between the unknown data point and all training data points
#     Sort the distance and find the nearest k data points
#     Classify the unknown data point based on the most instances of nearest k points

def kNN(k):
    # initialize the model with parameter k
    KNNClassifier = KNeighborsClassifier(n_neighbors=k)
    # train the model with X_train datapoints and y_train data labels
    KNNClassifier.fit(X_train, y_train)
    # returns a classification of a X_test datapoints
    labels = KNNClassifier.predict(X_test)
    
    return labels