# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:21:07 2020

@author: gilen
"""
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
#https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
df = pd.read_csv('winequality-red.csv') # Load the data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier

# The target variable is 'quality'.
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
X_featurenames = X.columns

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
# Build the model with the random forest regression algorithm:
clf = MLPClassifier(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=(100, 100), random_state=1, max_iter = 10000)

clf.fit(X_train, Y_train)


Y_Predict = clf.predict(X_test)

X_train["Predict"] = Y_Predict

X_test["Test"] = Y_test

X_test["Predict"] = Y_Predict

X_test["Error"] = abs(X_test["Test"] - X_test["Predict"])

X_test["Qtd Error"] = 1 if X_test["Test"] != X_test["Predict"] else 0

sum(X_test["Error"])

sum(X_test["Qtd Error"])