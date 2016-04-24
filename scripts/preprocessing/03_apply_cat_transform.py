# !/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# This script handles the preprocessing of our data and more specifically the columns with categorical values.
# Given the training dataset, we are converting the categorical values to the respective
# conditional probability for output 1 given each of those values.

# Import Library & Modules
import pandas as pd  # data processing, .csv files I/O
from pprint import pprint as pp
import pickle

# Read csv file, seperated by ',' 'nan' values exist
# The result is of pandas's type Dataframe. It is a table that consists columns and rows.
data = pd.read_csv('../../dataset/train.csv', sep=',', na_values='.')
test_data = pd.read_csv('../../dataset/test.csv', sep=',', na_values='.')

with open("../../dataset/02_transform_matrix.p", 'rb') as f:
    transform_matrix = pickle.load(f)

# Keep the code below for debugging reasons:

# for feature, values in transform_matrix.iteritems():
#     print(feature + " - before: ")
#     print(data[feature])
#
#     new_repl = {
#         feature: values
#     }
#     print("New replication matrix : ")
#     print(new_repl)
#
#     numerical_data = data.replace(new_repl)
#
#     print(feature + " - after: ")
#     print(numerical_data[feature])

# replace categorical values using the transformation matrix
# save the new dataframe in a pickle object

print("\n TRAINING DATA SET AFTER CHANGING CATEGORICAL TO NUMERICAL \n")
data = data.replace(transform_matrix)
pp(data)

with open('../../dataset/03_transformed_tr_dataframe.p', 'wb') as handle:
    pickle.dump(data, handle)

print("\n TEST DATA SET AFTER CHANGING CATEGORICAL TO NUMERICAL \n")
test_data = test_data.replace(transform_matrix)
pp(test_data)

with open('../../dataset/03_transformed_ts_dataframe.p', 'wb') as handle:
    pickle.dump(test_data, handle)
