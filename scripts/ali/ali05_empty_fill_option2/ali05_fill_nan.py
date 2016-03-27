
# coding: utf-8

# In[4]:

"Get Packages"
import numpy as np 
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
get_ipython().magic('matplotlib inline')


# In[8]:

"Get Data"
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[9]:

"Fill NaN values"
for f in train_df.columns:
    # fill NaN values withm -1
    if train_df[f].dtype == 'float64':
        train_df.loc[:,f][np.isnan(train_df[f])] = -1
        test_df[f][np.isnan(test_df[f])] = -1
        
    # fill NaN values with -1
    elif train_df[f].dtype == 'object':
        train_df[f][train_df[f] != train_df[f]] = -1
        test_df[f][test_df[f] != test_df[f]] = -1
        
for f in train_df.columns:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[f].values)  + list(test_df[f].values)))
        train_df[f]   = lbl.transform(list(train_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))


# In[11]:

sns.set_style('whitegrid')
plt.rcParams['figure.max_open_warning']=300
colnames=list(train_df.columns.values)
for i in colnames[2:]:
        facet = sns.FacetGrid(train_df, hue="target",aspect=2)
        facet.map(sns.kdeplot,i,shade= False)
        facet.add_legend()


# In[ ]:




# In[ ]:




# In[ ]:



