# coding: utf-8
__author__ = 'Antonis'

'''
Python script for Kaggle Competition

This script focuses on reading the train dataset and getting information for non-numerical values.
To run the script you need to have the "train.csv" file inside the /dataset folder in the project root.
'''

import pandas as pd

# Get the data from reading the training set
data = pd.read_csv('../dataset/train.csv', sep=',', na_values='.') #read csv file, seperated by ;, na values exists

# Find and print the names of all non-numerical features.
print("Export the features with non-numerical data")

non_numerical = []

# data is of pandas's type Dataframe. It is a table that consists columns and rows.
for column in data:
	# column_series is of pandas type Series ( One-dimensional ndarray with axis labels)
	column_series = data[column]

	# dtype is property of a Series. It declares the data type of the values inside it.
	if column_series.dtype not in ['int64', 'float64']:
		first_item = column_series.iloc[0]
		# Detect NaN values(they are calculated as Objects with type "float")
		if type(first_item) is not float:
			non_numerical.append(column)
print("")
print(non_numerical)
