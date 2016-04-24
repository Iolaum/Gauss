# !/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# This script handles the preprocessing of our data and more specifically the columns with categorical values.
# Given the training dataset, we are converting the categorical values to the respective
# conditional probability for output 1 given each of those values.

# Import Library & Modules
import pickle
from pprint import pprint as pp

with open("../../dataset/03_transformed_tr_dataframe.p", 'rb') as f:
    training_data = pickle.load(f)

# Get the median of each column(feature) from the training dataset
print("\n MEDIAN OF DATA \n")

median_of_data = training_data.median()
print(median_of_data)

# Apply the median of each feature to the NaN values of the training set
# ID & Target columns do not have NaN, so they are practically "skipped"
# Save new tr set into a pickle file

print("\n TRAINING DATA SET BEFORE FILLING NAN \n")
pp(training_data)

training_data = training_data.fillna(median_of_data)

print("\n TRAINING DATA SET AFTER FILLING NAN \n")
pp(training_data)

with open("../../dataset/04_tr_filled_data.p", 'wb') as f:
    pickle.dump(training_data, f)
del training_data

# Apply the median of each feature to the NaN values of the test set
# Save new ts set into a pickle file

with open("../../dataset/03_transformed_ts_dataframe.p", 'rb') as f:
    test_data = pickle.load(f)

# Clean non-test data features because fillna considers them as NaN and adds them
del median_of_data['target']

print("\n TEST DATA SET BEFORE FILLING NAN \n")
pp(test_data)

test_data = test_data.fillna(median_of_data)

print("\n TEST DATA SET AFTER FILLING NAN \n")
pp(test_data)

with open("../../dataset/04_ts_filled_data.p", 'wb') as f:
    pickle.dump(test_data, f)
