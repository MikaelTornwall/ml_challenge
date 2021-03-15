import numpy as np
import pandas as pd


# Data cleaning
# - Delete rows where y is empty / irrelevant
# - Delete/mean rows with fields where empty value


# Things to consider in each col:
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

    # x6 -  change misspelled labels: 'Bayesian Interference' -> 'Bayesian Inference', drop rows with 'nan' value
    unique_labels_x6 = pd.unique(data.x6)
    data = data[data.x6.notna()]
    data['x6'].replace({f'{unique_labels_x6[2]}': f'{unique_labels_x6[1]}'}, inplace=True)    
    data['x6'].replace({f'{unique_labels_x6[0]}': 0, f'{unique_labels_x6[1]}': 1}, inplace=True)    
    
    # x12 chage misspelled labels, 'Flase' -> 'False', encode the values: 
    unique_labels_x12 = pd.unique(data.x12) 
    data['x12'].replace({f'{unique_labels_x12[2]}': f'{unique_labels_x12[1]}'}, inplace=True)                
    data['x12'].replace({f'{unique_labels_x12[0]}': 0, f'{unique_labels_x12[1]}': 1}, inplace=True)                        
    
    X = data.iloc[:, 1:13].to_numpy()    
    y = data['y'].to_numpy()    

    # print(data)
    # print(unique_labels_x6)
    # print(unique_labels_x12)
    # print(pd.unique(data.x6))
    # print(pd.unique(data.x12))
    # data.info()
    
    return X, y


def getData(filename):            
    return pd.read_csv(filename, usecols=range(1,14))
