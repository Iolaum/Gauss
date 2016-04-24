# !/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# This script handles the preprocessing of our data and more specifically the columns with categorical values.
# Given the training dataset, we are converting the categorical values to the respective
# conditional probability for output 1 given each of those values.

# Import Library & Modules
import pickle
from pprint import pprint as pp

with open("../../dataset/03_transformed_dataframe.p") as f:
    training_data = pickle.load(f)

median_of_data = training_data.median()
print(median_of_data)

with open("../../dataset/04_tr_data_median.p", 'wb') as f:
    pickle.dump(median_of_data, f)

# ID & Target columns do not have NaN, so they are practically "skipped"
training_data = training_data.fillna(median_of_data)
pp(training_data)

with open("../../dataset/04_tr_filled_data.p", 'wb') as f:
    pickle.dump(training_data, f)
