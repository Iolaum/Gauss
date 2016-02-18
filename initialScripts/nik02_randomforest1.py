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
data = pd.read_csv('../dataset/train.csv', sep=',', na_values='.')
test = pd.read_csv('../dataset/test.csv')

# Create train predictor and train sets

# train = data[v1, ... v131]
# pred = data[target]

# pd.Series([1,2,3,4,'.']).convert_objects(convert_numeric=True)

pred = data['target'].convert_objects(convert_numeric=True)

del data['target']
del data['ID']

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
for column in data:
	print(column)
    # column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    column_series = data[column]

    # dtype is property of a Series. It declares the data type of the values inside it.
    if column_series.dtype not in ['int64', 'float64']:
        # print(type(column_series))
        le.fit(column_series)
        data[column] = le.transform(column_series)

        # dummy = le.transform(column_series)
        # print("")
        # print(column)
        # print(np.unique(dummy))

    # print(enc.transform(column_series))
    
    for index,value in column_series.iteritems():
    	if value == "":
    		column_series[index] = described_data[column]['mean']
    	

# use RandomForestRegressor for regression problem
# Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# class sklearn.preprocessing.OneHotEncoder(n_values='auto', categorical_features='all', dtype=<type 'numpy.float64'>, sparse=True, handle_unknown='error')

# Create Random Forest object
model = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets and check score
print("\nStarted Random Forest Training!")
model.fit(data, pred)

# Predict Output
predicted = model.predict(test)

print(type(predicted))

with open('results.txt', 'w') as f:
    for row in predicted:
        f.write(row)
