#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')


# In[2]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().system('pip install imbalanced-learn')


# In[4]:


df = pd.read_csv('bank.csv')


# In[5]:


df.dropna()


# In[6]:


x = df


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


le_job = LabelEncoder()
le_marital = LabelEncoder()
le_education = LabelEncoder()
le_default = LabelEncoder()
le_contact = LabelEncoder()
le_housing = LabelEncoder()
le_loan = LabelEncoder()
le_month= LabelEncoder()
le_poutcome = LabelEncoder()
le_deposit = LabelEncoder()


# In[9]:


x['job_n'] = le_job.fit_transform(x['job'])
x['marital_n'] = le_marital.fit_transform(x['marital'])
x['education_n'] = le_education.fit_transform(x['education'])
x['default_n'] = le_default.fit_transform(x['default'])
x['contact_n'] = le_contact.fit_transform(x['contact'])
x['housing_n'] = le_housing.fit_transform(x['housing'])
x['loan_n'] = le_loan.fit_transform(x['loan'])
x['month_n'] = le_month.fit_transform(x['month'])
x['poutcome_n'] = le_poutcome.fit_transform(x['poutcome'])
x['deposit_n'] = le_deposit.fit_transform(x['deposit'])
x


# In[10]:


x = x.drop(['job', 'marital', 'education', 'default','housing','contact','loan', 'month','poutcome', 'deposit',],axis = 'columns')
x


# In[11]:


corr_matrix = x.corr()
corr_matrix


# In[12]:


plt.figure(figsize=[20,20])
heat = sns.heatmap(corr_matrix,annot = True ,square = True)
heat


# In[13]:


x.head()


# In[14]:


x.shape


# In[17]:


y = x['loan_n']
y.shape


# In[18]:


x.shape


# In[19]:


x = x.drop('loan_n',axis='columns')
x.head()


# In[20]:


x.shape


# In[21]:


import imblearn
print(imblearn.__version__)


# In[22]:


from imblearn.under_sampling import NearMiss
undersample = NearMiss(sampling_strategy='auto',version=1,n_neighbors=2)
x,y = undersample.fit_resample(x, y)


# In[23]:


x.shape


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[25]:


from sklearn import tree


# In[26]:


model = tree.DecisionTreeClassifier( criterion='gini',splitter='best',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,min_impurity_decrease=0.0,class_weight=None,ccp_alpha=0.0,)
model.fit(x_train,y_train)


# In[27]:


model.predict(x_test)


# In[28]:


model.score(x_test,y_test)


# In[29]:


y_pred = model.predict(x_test)


# In[30]:


from sklearn.metrics import classification_report
cr = classification_report(y_pred,y_test)
cr


# In[31]:


from sklearn.ensemble import RandomForestClassifier
clff = RandomForestClassifier(n_estimators=1000,criterion='gini',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features='auto',max_leaf_nodes=None,min_impurity_decrease=0.0,bootstrap=True,oob_score=False,n_jobs=None,random_state=None,verbose=0,warm_start=False,class_weight=None,ccp_alpha=0.0,max_samples=None,)
clff.fit(x_train,y_train)


# In[32]:


clff.predict(x_test)


# In[33]:


clff.score(x_test,y_test)


# In[34]:


y_pred = clff.predict(x_test)


# In[35]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score
cm = confusion_matrix(y_pred,y_test)
cm


# In[36]:


from sklearn.metrics import classification_report
cr = classification_report(y_pred,y_test)
cr


# In[99]:


from xgboost import XGBClassifier
model = XGBClassifier( n_estimators=100,max_depth=2)
model.fit(x_train,y_train)


# In[100]:


y_pred=model.predict(x_test)


# In[101]:


model.score(x_test,y_test)


# In[123]:


from sklearn.metrics import classification_report
creport = classification_report(y_pred,y_test)
creport


# In[124]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score
cm = confusion_matrix(y_pred,y_test)
cm


# In[125]:


from sklearn.ensemble import AdaBoostClassifier


# In[126]:


booster = AdaBoostClassifier( n_estimators=150,learning_rate=0.5)
booster.fit(x_train,y_train)


# In[127]:


y_pred = booster.predict(x_test)


# In[128]:


booster.score(x_test,y_test)


# In[129]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score
cm = confusion_matrix(y_pred,y_test)
cm


# In[130]:


from sklearn.metrics import classification_report
creport = classification_report(y_pred,y_test)
creport


# In[131]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,n_estimators=100,)
# loss function = deviance(default) used in Logistic Regression
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)


# In[132]:


clf.score(x_test,y_test)


# In[133]:


from sklearn.metrics import confusion_matrix,precision_score,recall_score
cm = confusion_matrix(y_pred,y_test)
cm


# In[134]:


from sklearn.metrics import classification_report
creport = classification_report(y_pred,y_test)
creport


# In[ ]:




