#!/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# This script handles the preprocessing of our data and more specifically the columns with categorical values.

# Import Library & Modules
import pandas as pd  # data processing, .csv files I/O

# Read csv file, seperated by ',' 'nan' values exist
# The result is of pandas's type Dataframe. It is a table that consists columns and rows.
data = pd.read_csv('../../dataset/train.csv', sep=',', na_values='.')
#test = pd.read_csv('../../dataset/test.csv')

del data['target']
del data['ID']
# del test['ID']

# Populate 2 lists with all the columns that have categorical and numerical values.
cat_cols = []
num_cols = []

for idx, column in data.iteritems():
    # column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    column_series = data[idx]

    # dtype is property of a Series. It declares the data type of the values inside it.
    if column_series.dtype not in ['int64', 'float64']:
        # print(idx)
        # print(column.dtype)
        # consistency check because of nan is not considered numerical
        # print(column_series.head(10))

        # if the column has categorical values, print and add to the list
        cat_cols.append(idx)
    else:
        num_cols.append(idx)
        
        
print("Categorical Values are:")
print(cat_cols)
print("Numerical Values are:")
print(num_cols)
