# Python Script for Kaggle Competition
# BNP Paribas Cardif claim management

# testing
# print("Hello World!")

# modules needed
import numpy as np # linear algebraic manipulation
import csv # read data

#

with open('train.csv', 'r') as data:
    row = csv.DictReader(data, ',')
    for entry in row:
        print(entry)
        break
