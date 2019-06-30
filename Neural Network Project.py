import pandas as pd
import numpy as np
from os import path
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import eli5
from eli5.sklearn import PermutationImportance


def filetrain(ftrain):
    # Sets outputs as global variables to be called by other functions
    global X_train
    global y_train
    global UserTarget
    global final
    global TargetCol
    global features
    global scalerX
    global scalery
    global inum
    global con1
    global con2
    # Reads input file for training and stores it in variable
    ft = pd.read_csv(ftrain)
    # Runs pandas script to change num numerical values to dummy values (e.g., One Hot Encoding) and create a column for each categorical value
    # Drops first column of dummy values to avoid multicollinearity
    final = pd.get_dummies(ft, drop_first=True)
    # Adds NaN to blank cells
    final = final.fillna(0)
    UserTarget = input('What Column Label is your Target? ')
    TargetCol = final.columns.get_loc(UserTarget)
    # Stores data features (e.g, height and weight) from dataframe in array
    X_train = final.drop(UserTarget, axis='columns')
    # Stores feature names
    features = X_train.columns
    inum = len(features)
    X_train = X_train.values[:, :]
    # Fill blanks spots with NaN
    X_train = np.nan_to_num(X_train)
    # Stores target values (e.g., gender) from dataframe in array
    y_train = final.values[:, TargetCol]
    # Reshapes array to be a column of target values
    y_train = y_train.reshape(-1, 1)
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalery = MinMaxScaler(feature_range=(0, 1))
    X_train = scalerX.fit_transform(X_train)
    y_train = scalery.fit_transform(y_train)
    con1 = scalery.scale_[0]
    con2 = scalery.min_[0]


def train():
    # Sets reg outputs as global variables to be called by other functions
    global nnet
    nnet = Sequential()
    nnet.add(Dense(50, input_dim=inum, activation='relu', bias=False))
    nnet.add(Dense(100, activation='relu', bias=False))
    nnet.add(Dense(50, activation='relu', bias=False))
    nnet.add(Dense(1, activation='linear', bias=False))
    nnet.compile(loss="mean_squared_error", optimizer='adam')
    nnet.fit(X_train, y_train, epochs=100, shuffle='true', verbose=1)


def filepredict(fpredict):
    # Sets X1 and fp Outputs as global variables to be called by other functions
    global X_test
    global y_test
    global fp
    global TargetCol
    # Reads input file for prediction and stores it in variable
    fp = pd.read_csv(fpredict)
    fp = fp.fillna(0)
    fp = pd.get_dummies(fp, drop_first=True)
    y_test = fp.values[:, TargetCol]
    y_test = y_test.reshape(-1, 1)
    y_test = scalery.transform(y_test)
    # Identifies missing columns in test data
    missing_cols = set(features) - set(fp.columns)
    # Identifies extra columns in test data
    extra_cols = set(fp.columns) - set(features)
    # Removes extra columns
    for d in extra_cols:
        fp = fp.drop(d, axis='columns')
    # Add a missing column in test set with default value equal to 0 in the same location as the training file
    for c in missing_cols:
        loc = features.get_loc(c)
        if loc <= len(fp.columns):
            fp.insert(loc, c, 0, allow_duplicates=False)
        else:
            fp[c] = 0
    # Initiates loop if column names in test file are not in the same order as training file
    while all(fp.columns != features):
        # creates array out of column names in order
        colsort = np.array(features)
        # reshapes array
        colsort = colsort.reshape(1, -1)
        # reindexes prediciton file data in order of training data columns
        fp = fp.reindex(colsort, axis='columns')
    # Stores values in an array
    X_test = fp.values[:, :]
    X_test = scalerX.transform(X_test)


def predict():
    # Sets prediciton list as a global variable to be called by other functions
    global list
    # Initializes list as empty
    list = nnet.predict(X_test)
    list = (list - con2) / con1
    return list


def scorepredict():
    global features
    scorep = nnet.evaluate(X_test, y_test, verbose=0)
    print(scorep*100, "% Accurate")
    

def output():
    global y_test
    # Converts list array into a pandas Dataframe
    y_test = (y_test - con2) / con1
    y_test = pd.DataFrame(y_test)
    data = pd.DataFrame(list)
    # Names the resulting column
    data.columns = ['{} Guess'.format(UserTarget)]
    y_test.columns = ['{} Actual'.format(UserTarget)]
    # Merges the resulting predictions with the original input dataframe
    mergedres = pd.concat([fp, data, y_test], 1)
    # Outputs dataframe into a CSV file
    mergedres.to_csv(
        r'C:\Users\txz\Dropbox\Python Projects\Machine Learning Modules\sales_data_results.csv', index=False)


def main():
    output()


# Calls training file
ftrain = path.expanduser(
    r'C:\Users\txz\Dropbox\Python Projects\Machine Learning Modules\sales_data_training.csv')
filetrain(ftrain)
print(ftrain)
train()
# Calls prediction file
fpredict = path.expanduser(
    r'C:\Users\txz\Dropbox\Python Projects\Machine Learning Modules\sales_data_test.csv')
filepredict(fpredict)
predict()
scorepredict()

if __name__ == "__main__":
    main()
