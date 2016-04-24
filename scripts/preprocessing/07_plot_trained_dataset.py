# coding: utf-8
"Get Packages"
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
get_ipython().magic('matplotlib inline')

"Get Data"
with open('../../dataset/04_tr_filled_data.p', 'rb') as pyf:
    train_df = pickle.load(pyf)

sns.set_style('whitegrid')
plt.rcParams['figure.max_open_warning'] = 300
colnames = list(train_df.columns.values)
for i in colnames[2:]:
    facet = sns.FacetGrid(train_df, hue="target", aspect=2)
    facet.map(sns.kdeplot, i, shade=False)
    facet.add_legend()
