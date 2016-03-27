
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

import matplotlib.pyplot as plt
import numpy as np #linear algebra
import pandas as pd #data processing


# In[3]:

"Read Dataset"
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[4]:

"Check the shape of train and test data"
print(train.shape) #train shape
print(test.shape) #test shape


# In[5]:

"Find target variables"
train.target.value_counts().plot.bar() #target variable


# In[6]:

"Find the percentage of null values"
float(len(train[pd.isnull(train)]))/float((train.shape[1])*train.shape[0])


# In[7]:

"Null Value Percentage"
nullvalues = [float((train[col].isnull().sum()))/len(train[col])
              for col in train.columns.values]
percentagenull = list(zip(train.columns.values, nullvalues))
nullplot=pd.DataFrame(data=percentagenull,columns=["varname","percantage_null"])
nullplot=nullplot.set_index("varname")
nullplot.plot.bar(figsize =(23,5),title="percentage of null values per feature")


# In[8]:

"Find duplicate row in train set"
train.shape[0]-train.drop_duplicates().shape[0]


# In[9]:

"Find duplicate row in test set"
test.shape[0]-test.drop_duplicates().shape[0]


# In[10]:

"Constent feature count"
uniquecount=[train[col].nunique() for col in train.columns.values]
uniquecount=pd.DataFrame(data=list(zip(train.columns.values,uniquecount)),columns=["var","unique_count"])
unique_count=uniquecount[uniquecount.unique_count==1]
print("constent features count = {} ".format(unique_count.shape[0]))


# In[11]:

"seprating numeric and character features"
train_numr =train.select_dtypes(include=[np.number])
train_char =train.select_dtypes(include=[np.object])
print("Numerical column count : {}".format(train_numr.shape[1]))
print("Character column count : {}".format(train_char.shape[1]))


# In[12]:

"look at charcter features"
for col in  train_char:
    print(col+" : " +str(train_char[col].unique()[:10]))


# In[13]:

import random
df = train_numr.loc[np.random.choice(train_numr.index, 25000, replace=False)]
plt.matshow(df.corr())


# In[15]:

train = train.dropna()
targets = train['target']
indices = ['v' + str(i) for i in range(1,132,1)]
print(indices)
x = train[indices]
print(targets)
print(x)


# In[ ]:




# In[ ]:




# In[ ]:



