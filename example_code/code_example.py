#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:53:40 2023

@author: mingjunsun

This code is an example of solving LASSO-like optimization problems.
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn import linear_model
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV

# SET WORKING DIRECTORY
# data = pd.read_csv("../../usa.csv")
# first_rows = data.head(100)
# first_rows.to_csv("usa1.csv", index=False)
data = pd.read_csv("./example_code/usa1.csv")


# define start date and end date of training set
train_start = 19900000
train_end = 19911001

# define start date and end date of test set
test_start = 19911001
test_end = 20220000

# exclude data that's not during the period
data = data[(data["date"] > train_start) & (data["date"] < test_end)]

# exclude observations with missing market equity in month t and missing return in month t+1
data = data.dropna(subset=['me', 'ret_exc_lead1m'])

# exclude nano caps
data = data.loc[data['size_grp'] != 'nano']

# delete observation with more than 5 out of the 21 characteristics missing
cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d", "chcsho_12m",
        "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1", "at_be", "cash_at", "age", "z_score"]

data["missing_num"] = data[cols].isna().sum(1)
data = data.loc[data['missing_num'] <= 5]

# impute the missing characteristics by replacing them with the cross-sectional median
for i in cols:
    data[i] = data[i].astype(float)
    data[i] = data[i].fillna(data.groupby('date')[i].transform('median'))

cols1 = ["permno", "date", "ret_exc_lead1m", "be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d",
         "qmj_safety", "rmax1_21d", "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1",
         "at_be", "cash_at", "age", "z_score"]

data1 = data[cols1]
data1 = data1.dropna()

# rank transformation
# each characteristics is transformed into the cross-sectional rank
for i in cols:
    data1[i] = data1.groupby("date")[i].rank(pct=True)

# use df, store data1
df = data1
df.sort_values(by=['date', 'permno'], inplace=True)

# lists of months
train_month_list = df.date.unique()

df_test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
obj_month_list = df_test.date.unique()


# optimization part
def regularizer(beta):
    return cp.norm1(beta)


# def loss_fn(X, Y, beta, intercept):
#     return (1.0 / X.shape[0]) * (cp.norm2(X @ beta + intercept - Y)**2)
def loss_fn(X, Y, beta, intercept):
    if X.shape[0] == 0:
        return 0
    return (1.0 / X.shape[0]) * (cp.norm2(X @ beta + intercept - Y)**2)

# objective function 1 - L1 penalty on the coefficient


def objective_fn_initial(X, Y, beta, intercept, lambda1):
    return loss_fn(X, Y, beta, intercept) + lambda1 * cp.norm1(beta)


# coefficients, intercepts and objective values
beta_list = np.zeros((len(cols), 3))
intercept_list = np.zeros(3)
objective_list = np.zeros(3)

# set your own penalty parameters
lambda1_value = 0.0005

index = list(train_month_list).index(obj_month_list[0])
data_train = df[(df["date"] < train_month_list[index]) &
                (df["date"] >= train_month_list[index-5])]
X_train = data_train[cols].to_numpy()
Y_train = data_train["ret_exc_lead1m"].to_numpy()

# sklearn
model_lasso = linear_model.Lasso(alpha=lambda1_value/2)
model_lasso.fit(X_train, Y_train)
beta_list[:, 0] = model_lasso.coef_
intercept_list[0] = model_lasso.intercept_
objective_list[0] = np.linalg.norm(model_lasso.coef_, ord=1)

# cvx with objective function 1
beta = cp.Variable(len(cols))
intercept = cp.Variable(1)
problem = cp.Problem(cp.Minimize(objective_fn_initial(
    X_train, Y_train, beta, intercept, lambda1_value)))
problem.solve(solver=cp.SCS)
#beta_list[:,2] = beta.value
# Choose your own threshold here. I chose 0.0001.
beta_list[:, 1] = np.where(abs(beta.value) < 0.0001, 0, beta.value)
intercept_list[1] = intercept.value
objective_list[1] = problem.value
objective_list[1] = np.linalg.norm(beta.value, ord=1)


# cv
#! don't use this cv function
model_lasso_cv = LassoCV(alphas=[lambda1_value], cv=5)
model_lasso_cv.fit(X_train, Y_train)
beta_list[:, 2] = model_lasso_cv.coef_
intercept_list[2] = model_lasso_cv.intercept_
objective_list[2] = np.linalg.norm(model_lasso_cv.coef_, ord=1)

# The estimates in each column should be very close. Because there are the same model when lambda2_value = 0
# Feel free to try different penalty parameters and write your own optimization problems.
print(objective_list)
print(intercept_list)


# test
# process test data
data_test = df[(df["date"] <= obj_month_list[-1]) &
               (df["date"] >= obj_month_list[index])]
test_months = obj_month_list
print("test months ", test_months)
print("data_test ", data_test)

# prediction
true_Y, objective_sklearn, objective_cvxpy, objective_lasso_cv = [], [], [], []
for month in test_months:
    X_test = data_test[cols].to_numpy()
    Y_test = data_test["ret_exc_lead1m"].to_numpy()

    # TODO: append true y values of true_Y
    objective_sklearn.append(objective_fn_initial(
        X_test, Y_test, beta_list[:, 0], intercept_list[0], lambda1_value))
    objective_cvxpy.append(objective_fn_initial(
        X_test, Y_test, beta_list[:, 1], intercept_list[1], lambda1_value))
    objective_lasso_cv.append(objective_fn_initial(
        X_test, Y_test, beta_list[:, 2], intercept_list[2], lambda1_value))


print("true y values: ", [n.astype(int) for n in true_Y])
print("objective_sklearn: ", [obj.value for obj in objective_sklearn])
print("objective_cvxpy: ", [obj.value for obj in objective_cvxpy])
print("objective_lasso_cv: ", [obj.value for obj in objective_lasso_cv])


# Plot
test_months_counter = list(range(len(test_months)))
test_months_str = [str(month)[:8] for month in test_months]


# Extract numerical values from CVXPY expressions
objective_sklearn_values = [obj.value for obj in objective_sklearn]
objective_cvxpy_values = [obj.value for obj in objective_cvxpy]
objective_lasso_cv_values = [obj.value for obj in objective_lasso_cv]


plt.figure(figsize=(10, 6))
plt.plot(test_months_counter, objective_sklearn_values,
         label='Scikit-Learn Lasso', marker='o')
plt.plot(test_months_counter, objective_cvxpy_values,
         label='CVXPY Lasso', marker='s')
plt.plot(test_months_counter, objective_lasso_cv_values,
         label='Scikit-Learn LassoCV', marker='^')
for i, Y_test in enumerate(true_Y):
    plt.plot(test_months_counter, Y_test,
             label=f'True Values {i}', alpha=0.7)

plt.xlabel('Test Months')
plt.ylabel('Objective Values and True Values')
plt.title('Objective and True Values Comparison')
plt.legend()
plt.grid()
plt.xticks(test_months_counter, test_months_str, rotation=45)
plt.tight_layout()
plt.show()
