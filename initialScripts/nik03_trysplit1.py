#!/usr/bin/env python

# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# Doesn't work!



# Import Library & Modules
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np  # linear algebraic manipulation
import pandas as pd  # data processing, .csv files I/O

output = []
destfile = open('../dataset/test_splitted.csv', 'w')
csv_writer = csv.writer(destfile)

with open('../dataset/test.csv', 'r') as f:
	reader = csv.reader(f)
	c = 0
	for row in reader:
		csv_writer.writerow(row)
		c += 1
		if(c>1000):
			break

destfile.close()
