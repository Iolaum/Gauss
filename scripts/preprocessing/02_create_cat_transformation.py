from __future__ import division
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

# test = pd.read_csv('../../dataset/test.csv')
# print(data)
# print(test)

# Create a dictionary that would have the format:
# {
#   v13: {
#       A:0.15, B:0.80
#   }
# }

total_cond_prob_matrix = dict()
for idx, column in data.iteritems():
    # column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
    column_series = data[idx]

    # Get the columns with categorical values
    # dtype is property of a Series. It declares the data type of the values inside it.
    if column_series.dtype not in ['int64', 'float64']:
        # print(idx)
        # print(column.dtype)
        # consistency check because of nan is not considered numerical
        # print(column_series.head(10))

        cond_prob_matrix = dict()
        counter = 0

        # For each category add 1 to output_1 given the category
        # or 1 to output_0 given the category, depending on the output value
        for category in column_series:
            output_value = data['target'][counter]
            # print("Counter - " + str(counter))
            # print("Cat Value - " + str(cat_value))
            # print("Output Value - " + str(output_value))

            if category not in cond_prob_matrix:
                cond_prob_matrix[category] = {
                    'output_1': 0,
                    'output_0': 0
                }

            if output_value == 1:
                cond_prob_matrix[category]['output_1'] += 1
            else:
                cond_prob_matrix[category]['output_0'] += 1
            counter += 1

        # Count total of distinct values for each categorical feature
        total_cond_prob_matrix[idx] = {'distinct_categories': len(cond_prob_matrix)}

        for categ, categ_class_sums in cond_prob_matrix.iteritems():
            # print(categ)
            # print(categ_class_sums)

            # Calculate the conditional probability for output:1 for each value of each feature
            total_cond_prob_matrix[idx][categ] = categ_class_sums['output_1'] / (
                categ_class_sums['output_1'] + categ_class_sums['output_0']
            )

with open('../../dataset/02_transform_matrix.p', 'wb') as handle:
  pickle.dump(total_cond_prob_matrix, handle)

pp(total_cond_prob_matrix)
