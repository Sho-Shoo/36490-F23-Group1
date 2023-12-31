#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:25:15 2023

@author: mingjunsun
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
#from py_function import objective_fn_initial

from sklearn import linear_model 
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV


# os.chdir('/Users/mingjunsun/Library/CloudStorage/Dropbox/23 Summer/Data')

#read in rf
rf = pd.read_csv("data/rf.csv",index_col=False)

#read in SP 500
sp500 = pd.read_csv("data/index_wrds.csv",index_col=False)
sp500 = sp500[(sp500["tic"] == "I0003")]

#Choose start date and end date here
start_date = '1995-1-1'
end_date = '2022-1-1'

rf = rf[['caldt','t30ret']]
rf['caldt'] = pd.to_datetime(rf['caldt'])  
rf_obj = rf[(rf['caldt'] >= start_date) & (rf['caldt'] < end_date)]


#PRCCM -- Price - Close (PRCCM)
sp500['datadate'] = pd.to_datetime(sp500['datadate'])  
sp_obj = sp500[(sp500['datadate'] >= start_date) & (sp500['datadate'] < end_date)]

cols = ['datadate','prccm']
sp_obj = sp_obj[cols]

sp_obj['open'] = sp_obj['prccm'].shift(1)
sp_obj['return'] = (sp_obj['prccm'] - sp_obj['open']) / sp_obj['open']

rf_obj.index = range(len(rf_obj.index))
sp_obj.index = range(len(sp_obj.index))

extracted_col = rf_obj["t30ret"]
sp_obj = sp_obj.join(extracted_col)

sp_obj['excess_return'] = sp_obj['return'] - sp_obj['t30ret']

excess_return = np.array(sp_obj['excess_return'])
risk_free_return = np.array(sp_obj['t30ret'])

# print("SP500 Mean Return: "+ str(np.mean(sp_obj["return"])))
# print("SP500 Mean Excess Return: "+ str(np.mean(sp_obj["excess_return"])))
# print("SP500 Sharpe: "+str(np.mean(sp_obj["excess_return"])/ np.std(sp_obj["excess_return"])))

with open('outputs/portfolio/sp500.pkl', 'wb') as f:
    pickle.dump(excess_return, f)

with open('outputs/portfolio/risk_free.pkl', 'wb') as f:
    pickle.dump(risk_free_return, f)

