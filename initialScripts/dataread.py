
# coding: utf-8

# In[10]:

import numpy #linear Algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import random
from datetime import datetime
from sklearn.gaussian_process import GaussianProcess as GP
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy.stats import norm
from math import exp, fabs, sqrt, log, pi
from sklearn.ensemble import RandomForestClassifier as RFC
rnd=57
maxCategories=20


# In[11]:

"Read data files"
import pandas as pd
data = pd.read_csv('train.csv', sep=',', na_values='.') #read csv file, seperated by ;, na values exists
data #show data


# In[7]:

get_ipython().magic('whos')


# In[14]:

import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
pd.DataFrame.describe(data)#for large dataframes use pandas.DataFrame.describe()


# In[15]:




# In[ ]:



