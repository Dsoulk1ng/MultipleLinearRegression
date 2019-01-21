# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:02:47 2018

@author: kishl
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# Importing the dataset
dataset=pd.read_csv('50_Startups.csv')

# Separating Independent and Dependent Variables
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values 

# Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x[:,3]=label.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(categorical_features= [3])  #categorical_features denote the index of column to be one hot encoded
x = onehot.fit_transform(x).toarray()

# avoiding Dummy Variable Trap
x = x[:,1:]

# Splitting into Test Set and Training Set  #Data is always encoded before splitting
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 0) 

# No reqirement for Feature Scalling because the Library takes care of it

# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
  
# Prediction                                <<<ALL IN>>>
y_pred = regressor.predict(x_test)

# Prediction                                <<<BACKWARD ELEMINATION>>>
import statsmodels.formula.api as sm
x = np.append(np.ones((50,1)).astype(int),values = x ,axis =1)