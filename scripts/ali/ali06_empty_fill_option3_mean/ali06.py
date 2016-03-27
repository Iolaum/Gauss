
# coding: utf-8

# In[4]:

"Get Packages"
import numpy as np #numpy package
import pandas as pd #pandas package
from pandas import Series, DataFrame #series, dataframe sub-package
import matplotlib.pyplot as plt #matplotlib for plotting
import seaborn as sns #for graph griding
sns.set_style('whitegrid')
from sklearn import preprocessing #preprocessing
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.svm import SVC, LinearSVC #linear single value classifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
get_ipython().magic('matplotlib inline')


# In[5]:

"Get Data"
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[6]:

"Fill NaN values"
for f in train_df.columns:
    # fill NaN values with mean
    if train_df[f].dtype == 'float64':
        train_df[f][np.isnan(train_df[f])] = train_df[f].mean()
        test_df[f][np.isnan(test_df[f])] = test_df[f].mean()
        
    # fill NaN values with most occured value
    elif train_df[f].dtype == 'object':
        train_df[f][train_df[f] != train_df[f]] = train_df[f].value_counts().index[0]
        test_df[f][test_df[f] != test_df[f]] = test_df[f].value_counts().index[0]
        
for f in train_df.columns:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[f].values)  + list(test_df[f].values)))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))


# In[7]:

plt.rcParams['figure.max_open_warning']=300
colnames=list(train_df.columns.values)
for i in colnames[2:]:
        facet = sns.FacetGrid(train_df, hue="target",aspect=2)
        facet.map(sns.kdeplot,i,shade= False)
        facet.add_legend()


# In[8]:

"Define Training and Testing set"
x_train = train_df.drop(["ID","target"],axis=1)
y_train = train_df["target"]
x_test = test_df.drop("ID",axis=1).copy()


# In[10]:

"Logistic Regression"
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict_proba(x_test)[:,1]
logreg.score(x_train, y_train)


# In[11]:

"Random Forest"
random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict_proba(x_test)[:,1]
random_forest.score(x_train, y_train)


# In[14]:

"Using Logistic Regression get coefficient"
coeff_df = DataFrame(train_df.columns.delete([0,1]))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = (pd.Series(logreg.coef_[0]))** 2
coeff_df.head() #preview top five


# In[15]:

"Plot coefficient of determination in order"
coeff_ser = Series(list(coeff_df["Coefficient Estimate"]),
                  index = coeff_df["Features"]).sort_values()
fig = coeff_ser.plot(kind = 'barh', figsize = (15,5))
fig.set(ylim = (116, 131))


# In[16]:

"Submission"

submission = pd.DataFrame()
submission["ID"] = test_df["ID"]
submission["PredictedProb"] = y_pred
submission.to_csv('ali06_mean_predict.csv', index=False)


# In[ ]:



