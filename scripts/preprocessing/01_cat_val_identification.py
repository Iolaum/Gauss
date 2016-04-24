from __future__ import division
# !/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# This script handles the preprocessing of our data and more specifically the columns with categorical values.

# Import Library & Modules
import pandas as pd  # data processing, .csv files I/O
import pickle

# Read csv file, seperated by ',' 'nan' values exist
# The result is of pandas's type Dataframe. It is a table that consists columns and rows.
data = pd.read_csv('../../dataset/train.csv', sep=',', na_values='.')
del data['target']
del data['ID']

# Populate 2 lists with all the columns that have categorical and numerical values.
cat_cols = []
num_cols = []

for idx, column in data.iteritems():
    # column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    column_series = data[idx]

    # dtype is property of a Series. It declares the data type of the values inside it.
    # Get the columns with categorical values

    total_cond_prob_matrix = dict()

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
with open('../../dataset/01_cat_col.p', 'wb') as handle:
  pickle.dump(cat_cols, handle)
  
print("Numerical Values are:")
print(num_cols)
with open('../../dataset/01_num_col.p', 'wb') as handle:
  pickle.dump(num_cols, handle)
