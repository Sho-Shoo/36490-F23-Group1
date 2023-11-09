#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:02:34 2023

@author: mingjunsun
"""


import numpy as np
import pandas as pd
import os
#from py_function import objective_fn_initial


def get_top_10_percent_indices(arr):
    # Calculate the number of elements corresponding to the top 10%
    top_10_percent = int(0.1 * len(arr))
    
    # Create an array of indices from 0 to len(arr) - 1
    indices = np.arange(len(arr))
    
    # Shuffle the indices randomly
    np.random.shuffle(indices)
    
    # Use argsort to sort the shuffled indices based on the corresponding elements in arr
    sorted_indices = indices[np.argsort(arr[indices])]
    
    # Get the indices of the top 10% values
    top_10_percent_indices = sorted_indices[-top_10_percent:]
    
    return top_10_percent_indices

def get_bottom_10_percent_indices(arr):
    # Calculate the number of elements corresponding to the bottom 10%
    bottom_10_percent = int(0.1 * len(arr))
    
    # Create an array of indices from 0 to len(arr) - 1
    indices = np.arange(len(arr))
    
    # Shuffle the indices randomly
    np.random.shuffle(indices)
    
    # Use argsort to sort the shuffled indices based on the corresponding elements in arr
    sorted_indices = indices[np.argsort(arr[indices])]
    
    # Get the indices of the bottom 10% values
    bottom_10_percent_indices = sorted_indices[:bottom_10_percent]
    
    return bottom_10_percent_indices


# #read the coefficient estimates
# beta_list = pd.read_csv("/.csv",index_col=0)
# beta_list = np.array(beta_list)

# pred_true_table = pd.DataFrame()
# rets_ls_vw = np.zeros(len(obj_month_list))
# rets_l_vw = np.zeros(len(obj_month_list))


# #get each month's characteristics and next month's return
# #here, wecalculate the value-weighted portfolio returns in the first month of obj_month_list
# k = 0
# data_train = df[(df["date"] == obj_month_list[k])]
# X_train = data_train[cols].to_numpy()
# Y_train = data_train["ret_exc_lead1m"].to_numpy()
# size_train = data_train["market_equity1"].to_numpy()
        
# #predicted return using the coefficients in the beta_list
# coef = beta_list
# preds = X_train @ coef

# pred_true_combined = np.vstack((preds,Y_train,size_train)).T
# pred_true_combined = pd.DataFrame(pred_true_combined)
# pred_true_combined.columns = ["pred","true","size"]
# pred_true_combined['true_size'] = pred_true_combined[["true"]].multiply(pred_true_combined["size"], axis="index")
# pred_true_combined = np.array(pred_true_combined)

# num_stocks = pred_true_combined.shape[0] // 10
# highest_indices = get_top_10_percent_indices(pred_true_combined[:,0])
# lowest_indices = get_bottom_10_percent_indices(pred_true_combined[:,0])
   
# #value weighted portfolio return
# #rets_ls_vw is the excess return of the value-weighted long-short portfolio
# #rets_l_vw is the excess return of the value-weighted long portfolio
# rets_ls_vw[k] = 0.5*(np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2]) - 
#                      np.sum(pred_true_combined[lowest_indices,3])/np.sum(pred_true_combined[lowest_indices,2]))
# rets_l_vw[k] = np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2])





