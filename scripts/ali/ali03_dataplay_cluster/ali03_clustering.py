
# coding: utf-8

# In[4]:

import pandas as pd

train = pd.read_csv('train.csv') # Load train data
train = train.drop(['ID', 'target'], axis=1) #ID, target, not needed

test = pd.read_csv('test.csv')
test = test.drop(['ID'], axis=1) #ID not needed for analysis

data = train.append(test, ignore_index=True) #concat datasets
data.info()


# In[8]:

length = len(data) #full length of data

filled = [] #100% filled
empty = [] #not filled, not empty

x_int = [] #float+integer
y_int = [] #float+integer

x_str = [] #string
y_str = [] #string

for i, (name, series) in enumerate(data.iteritems()):
    c = series.count() #return count of filled rows
    fill = c/length*100
    
    if series.dtype == 'O': #print name, fill, series.dtype
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)
    
    if fill>80:
        filled.append(name) #above 80%
    else:
        empty.append(name)
            


# In[9]:

filled


# In[10]:

len(filled)


# In[11]:

empty


# In[12]:

len(empty)


# In[15]:

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
plt.subplots(figsize=(10, 10))
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)
plt.axhline(80,color='r') #80% treshold
plt.show()


# In[16]:

"build affinity matrix, calculate distance between each dataseries"
tmp_row1 = list(range(10))
tmp_row2 = list(range(10,20))
tmp_row2[2:4] = [None,None]
tmp_row3 = list(range(20,30))
tmp_row3[2:6] = [None,None,None,None]
example_df = pd.DataFrame(data={'v1':tmp_row1, 'v2':tmp_row2, 'v3':tmp_row3})
example_df


# In[17]:

import numpy as np
def dist(series1, series2, length):#correlation between data series
    c = series1.isnull().values == series2.isnull().values
    return np.sum(c.astype(int))/length


# In[18]:

dist(example_df['v1'],example_df['v2'],10)


# In[19]:

dist(example_df['v1'],example_df['v3'],10)


# In[20]:

dist(example_df['v1'],example_df['v2'],10)


# In[21]:

"Affinity matrix"
all_data = data.drop(filled, axis=1) #all data low filled
a = np.eye(len(empty))  #matrix already have 1 in main diagonal
length = len(all_data)

for i, (name1, series1) in enumerate(all_data.iteritems()):
    for j, (name2, series2) in enumerate(all_data.iteritems()):
        if j == i:  #under main diag
            break
        else:
            tmp_d = dist(series1, series2, length)
            a[i,j] = tmp_d
            a[j,i] = tmp_d
       
a.shape


# In[23]:

size = len(empty)
fig, ax = plt.subplots(figsize=(15, 15))
ax.matshow(a)
locs, labels = plt.xticks(range(size), empty)
plt.setp(labels, rotation=90)
plt.yticks(range(size), empty)
plt.show()


# In[24]:

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

x_score = []
y_score = []

for i in range(2,10): #Get scores for n_clusters from 2 to 10
    tmp_clf = SpectralClustering(n_clusters=i, affinity='precomputed')
    tmp_clf.fit(a)
    score = silhouette_score(a, tmp_clf.labels_, metric='precomputed')
    x_score.append(i)
    y_score.append(score)

plt.subplots(figsize=(10, 10))
plt.plot(x_score,y_score)
plt.grid()
plt.show()


# In[25]:

clusters_count = 3
clusters = [[] for i in range(clusters_count)]
clf = SpectralClustering(n_clusters=clusters_count, affinity='precomputed', random_state=42)
clf.fit(a)

for name,cluster_n in zip(empty, clf.labels_):
    clusters[cluster_n].append(name)
    
for tmp_cluster in clusters:
    print('---')
    print(tmp_cluster)


# In[26]:

clusters_count = 4
clusters = [[] for i in range(clusters_count)]
clf = SpectralClustering(n_clusters=clusters_count, affinity='precomputed', random_state=42)
clf.fit(a)

for name,cluster_n in zip(empty, clf.labels_):
    clusters[cluster_n].append(name)
    
for tmp_cluster in clusters:
    print('---')
    print(tmp_cluster)


# In[28]:

def get_mask_notnull(df,columns_list):
    i = iter(columns_list)
    #Take first column in list
    first_v = next(i)
    #Get notnull mask
    current_mask = df[first_v].notnull()
    for tmp_v in i:
        current_mask = current_mask | df[tmp_v].notnull() #logical "or"
    
    return current_mask

# Get elements from first cluster
df_first_cluster = all_data[get_mask_notnull(all_data, clusters[0])]
print('objects count from cluster 1: %d' % len(df_first_cluster))

# And draw filling percentage as in the beginning of script

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_first_cluster)
for i,(name,series) in enumerate(df_first_cluster.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)
    
plt.subplots(figsize=(10, 10))
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# In[29]:

# Get elements from combine (1+2) cluster
df_combine = all_data[get_mask_notnull(all_data, clusters[0]+clusters[1])]
print('objects count from clusters 1 and 2: %d' % len(df_combine))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_combine)
for i,(name,series) in enumerate(df_combine.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10)) 
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# In[30]:

# Get elements from 2 cluster
df_third = all_data[get_mask_notnull(all_data, clusters[2])]
print('objects count from cluster 3: %d' % len(df_third ))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_third )
for i,(name,series) in enumerate(df_third .iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10))         
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# In[31]:

df_high_filled = all_data[~get_mask_notnull(all_data, clusters[0]+clusters[1])]
print('only high filled objects: %d' % len(df_high_filled))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_high_filled)
for i,(name,series) in enumerate(df_high_filled.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10))         
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# In[ ]:



