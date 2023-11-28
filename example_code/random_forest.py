#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 01:26:44 2023

@author: mingjunsun
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
#from livelossplot import PlotLosses
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#set working directory
os.chdir('/Users/mingjunsun/Library/CloudStorage/Dropbox/23 Summer/Data/')

data_original = pd.read_csv("Characteristics/usa.csv")

data = data_original
data_expl = data.head(105)

#exclude data before 1991-12-31
data = data[~(data['date'] < 19920000)]

#exclude observations with missing me in month t and missing return in month t+1
data = data.dropna(subset=['me', 'ret_exc_lead1m'])

#exclude nano caps
data = data.loc[data['size_grp'] != 'nano']

#delete observation with more than 5 out of the 21 characteristics missing
cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m","ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be","cash_at", "age", "z_score"]
data["missing_num"] = data[cols].isna().sum(1)
data = data.loc[data['missing_num'] <= 5]

#impute the missing characteristics by replacing them with the cross-sectional median
for i in cols:
    data[i] = data[i].astype(float)
    data[i] = data[i].fillna(data.groupby('date')[i].transform('median'))



#check missing values of the 21 characteristics
#randomlist = random.sample(range(1, 310000), 150)
#data_expl = data.iloc[randomlist]
#cols1 = ["missing_num", "id", "date","be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m","ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be","cash_at", "age", "z_score"]
#data_expl = data_expl[cols1]


#predict next month's excess return over the training period

#OLS pytorch
cols1 = ["date","ret_exc_lead1m", "be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m","ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be","cash_at", "age", "z_score"]
data1 = data[cols1]
data1 = data1.dropna()

data_train = data1[(data1['date'] < 20120000)]
data_test = data1[(data1['date'] > 20120000)]

X_train_all = data_train[cols].to_numpy()
Y_train_all = data_train["ret_exc_lead1m"].to_numpy()

# split into training and validation
data_validation = data_train[(data_train['date'] > 20050000)]
data_train = data_train[(data_train['date'] < 20050000)]

X_train = data_train[cols].to_numpy()
Y_train = data_train["ret_exc_lead1m"].to_numpy()

X_test = data_test[cols].to_numpy()
Y_test = data_test["ret_exc_lead1m"].to_numpy()

X_validation = data_validation[cols].to_numpy()
Y_validation = data_validation["ret_exc_lead1m"].to_numpy()


P = [7,14,21]
D = [1,2,3]
M = [2,10]

p = 7
d = 1
m = 2

'''
for p in P:
    for d in D:
        for m in M:
            rf = RandomForestRegressor(n_estimators = 500, max_depth = d, min_samples_split = m, max_features = p, bootstrap = True, max_samples = 0.5)
# Train the model on training data
            rf.fit(X_train, Y_train)
            predictions = rf.predict(X_validation)
            mse_error = mean_squared_error(predictions, Y_validation)
# Print out the mean absolute error (mae)
            print('P=',p,' D=', d, ' M=', m ,'  Mean Squared Error:', round(mse_error, 6), 'degrees.')
'''

# we choose P =7, D = 1, M = 10


rf = RandomForestRegressor(n_estimators = 500, max_depth = 1, min_samples_split = 10, max_features = 7, bootstrap = True, max_samples = 0.5)
# Train the model on training data
rf.fit(X_train_all, Y_train_all)
predictions = rf.predict(X_test)
mse_error = mean_squared_error(predictions, Y_test)
print(mse_error)



