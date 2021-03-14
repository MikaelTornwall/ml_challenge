import numpy as np
import pandas as pd
from knn import *


# Models to try
#
# K-nearest neighbor
# Naive Bayes
# Linear Discriminative Analysis
# Quadratic Discriminative Analysis
# Logistic Regression
# Ensemble learning
#   Boosting
#   Bagging
#   Random Forests (suited for multi-class problems)
# SVM
# Multilayer Perceptron
#

def getData(filename):
    df = pd.read_csv(filename)
    head = df.head()    
    print(head)
    df.info()


def main():
    getData('TrainOnMe.csv')
    getData('EvaluateOnMe.csv')

if __name__ == "__main__":
    main()