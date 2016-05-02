# !/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# Import Library & Modules
import pickle
import numpy as np


# Transform categorical values of columns v22       v56       v71      v113
# that were not in the training dataset to NaN
# and OVERRIDES the file

def replace_cat(value):
    if type(value) not in [np.int64, np.float64, float]:
        value = np.nan
    return value


with open("../../dataset/03_transformed_ts_dataframe.p", 'rb') as f:
    test_set_df = pickle.load(f)

print("Check before transformation")
cat_data = test_set_df.select_dtypes(include=[np.object])
num_data = test_set_df.select_dtypes(include=[np.number])

print("Categorical data columns")
print(list(cat_data.columns.values))

print("Numerical data columns")
print(list(num_data.columns.values))

for column in test_set_df:
    test_set_df[column] = test_set_df[column].map(lambda x: replace_cat(x))

print("Check after transformation")
cat_data = test_set_df.select_dtypes(include=[np.object])
num_data = test_set_df.select_dtypes(include=[np.number])

print("Categorical data columns")
print(list(cat_data.columns.values))

print("Numerical data columns")
print(list(num_data.columns.values))

print("Saving test dataframe. Re-run scripts 04 and afterwards after that")
with open('../../dataset/03_transformed_ts_dataframe.p', 'wb') as handle:
    pickle.dump(test_set_df, handle)
