# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 19:05:47 2018

@author: kishl
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset=pd.read_csv("50_Startups.csv")
x = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values

# Ecoding the data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
x[:,3] = label.fit_transform(x[:,3])
from sklearn.preprocessing import OneHotEncoder 
onehot = OneHotEncoder(categorical_features=[3])
x = onehot.fit_transform(x).toarray()

# Avoiding Dummy Variable Trap
x=x[:,1:]

# Splitting into Test Set and Training Set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#Fitting Multiple Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Prediction of test results
y_pred = regressor.predict(x_test)

# Building Backward Elemination
import statsmodels.formula.api as sm

 x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

# Trail 1
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

# Trail 2
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

# Trail 3
x_opt = x[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

# Trail 4
x_opt = x[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

# Trail 5
x_opt = x[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog= x_opt).fit()
regressor_OLS.summary()

# Splitting  X_opt into Test Set and Training Set
from sklearn.cross_validation import train_test_split
x_train_opt,x_test_opt,y_train_opt,y_test_opt= train_test_split(x_opt,y,test_size=0.2,random_state=0)

#Fitting Multiple Regression Model on X_Optimum
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train_opt,y_train_opt)

# Prediction of test results
y_pred_opt = regressor.predict(x_test_opt)
