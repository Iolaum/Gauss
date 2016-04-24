#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # standardise data


import pickle
from sklearn.preprocessing import StandardScaler

with open('../../dataset/05_split_xtr.p', 'rb') as f:
    xtr = pickle.load(f)
print("loaded xtr, type is {} \n and shape is {}".format(type(xtr), xtr.shape))

xscaler = StandardScaler().fit(xtr)
print xscaler.mean_
print xscaler.scale_
