import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

#
# Cleaning
#
# y - 4 unique real labels, then empty ones and rows with random Icelandic(?) text
# x1 - x5, x7 - x11 check ranges, normalize(?) and check for 'nan':s and outliers
# -> use pandas interpolate to fill nan -values 
#
# x8 might contain outliers                
#
# x6 - 2 unique real labels ['Bayesian Inference', 'GMMs and Accordions'] then 'nan' and 'Bayesian Interference'
#   Transform 'Bayesian Interference' -> 'Bayesian Inference'
#
# x12 - 2 unique real labels [True, False], then 'nan' and 'Flase'
#   Transform 'Flase' -> 'False'
#


def cleanData(data):                
    # y
    data = data[data.y.notna()]    
    unique_labels_y = pd.unique(data.y)
    data = data[data.y != unique_labels_y[4]]
    data = data[data.y != unique_labels_y[5]]
    
    # x1 - drop rows with outliers
    threshold = 10**3
    data = data[data.x1 < threshold]
    data = data[data.x1 > -threshold]       

    # x6 -  change misspelled labels: 'Bayesian Interference' -> 'Bayesian Inference', drop rows with 'nan' value, encode the values
    # unique_labels_x6 = pd.unique(data.x6)
    # data = data[data.x6.notna()]
    # data['x6'].replace({f'{unique_labels_x6[2]}': f'{unique_labels_x6[1]}'}, inplace=True)    
    # data['x6'].replace({'Bayesian Inference': 0, 'GMMs and Accordions': 1}, inplace=True)    
    
    # x12 chage misspelled labels, 'Flase' -> 'False', encode the values: 
    # unique_labels_x12 = pd.unique(data.x12) 
    # data['x12'].replace({f'{unique_labels_x12[2]}': f'{unique_labels_x12[1]}'}, inplace=True)                
    # data['x12'].replace({'True': 1, 'False': 0}, inplace=True)                        
    # print(unique_labels_x12)

    # drop columns x2, x3, x4, x5, x6 and x12
    data.drop('x2', inplace=True, axis=1)
    data.drop('x3', inplace=True, axis=1)
    data.drop('x4', inplace=True, axis=1)
    data.drop('x5', inplace=True, axis=1)
    data.drop('x6', inplace=True, axis=1)
    data.drop('x12', inplace=True, axis=1)
    
    rows, cols = data.shape

    # normalize feature vectors
    normalizer = StandardScaler()
    X = normalizer.fit_transform(data.iloc[:, 1:cols])

    y = data['y'].to_numpy()    
    
    # feature selection
    # X = select_features(X, y)
    
    return X, y


def get_evaluation_data():
    data = getData('EvaluateOnMe.csv', 13)
    data.drop('x2', inplace=True, axis=1)
    data.drop('x3', inplace=True, axis=1)
    data.drop('x4', inplace=True, axis=1)
    data.drop('x5', inplace=True, axis=1)
    data.drop('x6', inplace=True, axis=1)
    data.drop('x12', inplace=True, axis=1)    
    normalizer = StandardScaler()    
    X = normalizer.fit_transform(data)
    return X


# supervised feature selection
# does not seem to enhance classification accuracy
def select_features(X, y):
    X_new = SelectKBest(f_classif, k=6).fit_transform(X, y)
    print(X_new.shape)
    return X_new


def getData(filename, cols=14):            
    return pd.read_csv(filename, usecols=range(1,cols))
