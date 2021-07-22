#!/usr/bin/env python
# coding: utf-8

# # Neural Network Assignment

# In[1]:


# import libraries
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('framingham.csv')
data.keys()
data.head()


# In[3]:


# Checking for missing data
data.isna().sum()


# In[6]:


data.dropna(axis=0, inplace = True)


# In[18]:


# Extracting feature vector X (all columns except the last) and output label y (last column)
y = data.values[:, -1]
X = data.values[:, :-1]


# In[19]:


# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)


# In[20]:


# Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)


# In[21]:


# Building neural network 3 layers
# with 11 nodes in the first layer, 58 nodes in the second layer, 
# and 90 node in the third layer
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(11,30,60), max_iter=1000)
mlp.fit(X_train,y_train)


# In[22]:


# Computing the train and test scores
print("test: ", mlp.score(X_test, y_test))
print("train: ",mlp.score(X_train, y_train))

