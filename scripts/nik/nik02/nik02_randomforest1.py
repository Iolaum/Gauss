#!/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# Doesn't work!



# Import Library & Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np  # linear algebraic manipulation
import pandas as pd  # data processing, .csv files I/O

# When checking unique, numpy handles this error so far.
import warnings

warnings.filterwarnings('ignore', 'numpy equal will not check object identity in the future')
warnings.filterwarnings('ignore', 'numpy not_equal will not check object identity in the future')

# read csv file, seperated by , na values exists
data = pd.read_csv('../dataset/train_splitted.csv', sep=',', na_values='.')
test = pd.read_csv('../dataset/test_splitted.csv')

# Create train predictor and train sets

# train = data[v1, ... v131]
# pred = data[target]

# pd.Series([1,2,3,4,'.']).convert_objects(convert_numeric=True)

pred = data['target'].convert_objects(convert_numeric=True)

del data['target']
del data['ID']
del test['ID']

# print(pred.ix[0])
# print(data.ix[0])


# Use describe data to get information from the table (count, mean, std, min, max values, etc)
described_data = pd.DataFrame.describe(data)
# print(described_data)

# get count of nan values in the columns
# null_nos_per_col = data.isnull().sum()
# print(null_nos_per_col)

le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()
imp = preprocessing.Imputer()

# data is of pandas's type Dataframe. It is a table that consists columns and rows.
for idx, column in data.iteritems():
    print(idx)
    # column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    column_series = data[idx]

    # dtype is property of a Series. It declares the data type of the values inside it.
    if column_series.dtype not in ['int64', 'float64']:
        # print(type(column_series))
        le.fit(column_series)

        column_series = le.transform(column_series)
        data.loc[:, idx] = column_series

        # dummy = le.transform(column_series)
        # print("")
        # print(column)
        # print(np.unique(dummy))

    # fill nan values on each series for each row
    # else:
    #     column_series_mean = described_data[idx]['mean']
    #     for index, value in column_series.iteritems():
    #         if pd.isnull(value):
    #             data.loc[index, idx] = column_series_mean

    # fill nan values inside the whole table
    data = data.fillna(data.mean())

# use RandomForestRegressor for regression problem
# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# class sklearn.preprocessing.OneHotEncoder(n_values='auto', categorical_features='all', dtype=<type 'numpy.float64'>, sparse=True, handle_unknown='error')

data_two = data.astype(np.float32)
# print(data_two.dtypes)
# print(data_two.isnull().sum())

# Create Random Forest object
model = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets and check score
print("\nStarted Random Forest Training!")
model.fit(data_two, pred)

# data is of pandas's type Dataframe. It is a table that consists columns and rows.
for idx, test_col in test.iteritems():
    print(idx)
    # test_col_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    test_col_series = test[idx]

    # dtype is property of a Series. It declares the test type of the values inside it.
    if test_col_series.dtype not in ['int64', 'float64']:
        # print(type(test_col_series))
        le.fit(test_col_series)

        test_col_series = le.transform(test_col_series)
        test.loc[:, idx] = test_col_series

        # dummy = le.transform(test_col_series)
        # print("")
        # print(test_col)
        # print(np.unique(dummy))

    # fill nan values on each series for each row
    # else:
    #     test_col_series_mean = described_test[idx]['mean']
    #     for index, value in test_col_series.iteritems():
    #         if pd.isnull(value):
    #             test.loc[index, idx] = test_col_series_mean

    # fill nan values inside the whole table
    test = test.fillna(test.mean())

# Predict Output
predicted = model.predict(test)

print(type(predicted))

with open('../dataset/results.txt', 'w') as f:
    for row in predicted:
        f.write(str(row))
