#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


# In[19]:


iris_data = load_iris()
iris_data


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=121)


# In[13]:


dtree = DecisionTreeClassifier()
dtree


# In[14]:


parameters = {'max_depth':[1,2,3],
                   'min_samples_split':[2,3]}


# In[15]:


import pandas as pd

grid_tree = GridSearchCV(dtree, param_grid=parameters, cv=3, refit=True)


# In[21]:


grid_tree.fit(X_train, y_train)


# In[23]:


grid_tree.cv_results_


# In[24]:


scores_df = pd.DataFrame(grid_tree.cv_results_)


# In[25]:


scores_df


# In[26]:


scores_df[['params','mean_test_score','rank_test_score','split0_test_score']]


# In[29]:


print('GridSerachCV 최적 파라미터: ', grid_tree.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_tree.best_score_))


# In[ ]:




