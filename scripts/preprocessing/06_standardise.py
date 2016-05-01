#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # standardise data


import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

with open('../../dataset/05_split_xtr.p', 'rb') as f:
    xtr = pickle.load(f)
# print("loaded xtr, type is {} \n and shape is {}".format(type(xtr), xtr.shape))

print("Creating Standardizer")
xscaler = StandardScaler().fit(xtr)
# Debug
# print xscaler.mean_.shape
# print xscaler.scale_.shape

new_arr = np.column_stack((xscaler.mean_, xscaler.scale_))

#
# print("Check if we need to standardize")
# print(new_arr)
#
# print("Standardize")
# standardized_xtr = xscaler.transform(xtr)
# print(standardized_xtr)
#
# print("verification")
# xscaler_two = StandardScaler().fit(standardized_xtr)
# new_arr = np.column_stack((xscaler_two.mean_, xscaler_two.scale_))
# print(new_arr)

with open('../../dataset/06_standardizer.p', 'wb') as f:
    pickle.dump(xscaler, f)
    print("Standardizer saved!")
