import pickle
import numpy as np
import pandas as pd
from example_code.portfolio_code import get_top_10_percent_indices
from example_code.portfolio_code import get_bottom_10_percent_indices
import matplotlib.pyplot as plt
from tqdm import tqdm

cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d",
            "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1",
            "at_be", "cash_at", "age", "z_score"]

def loadData():
    data = pd.read_csv('data/usa_short.csv')

    # drop missing observations
    data = data.dropna(subset=['me','ret_exc_lead1m','permno'])

    # exclude nano caps
    data = data[data['size_grp'] != 'nano']

    # delete observation with more than 5 out of the 21 characteristics missing
    data["missing_num"] = data[cols].isna().sum(axis=1)
    data = data[data['missing_num'] <= 10]

    # impute the missing characteristics by replacing them with the cross-sectional median
    for i in cols:
        data[i] = data[i].fillna(data.groupby('date')[i].transform('median'))


    # rank transformation following Gu-Kelly-Xiu 2020 RFS
    # each characteristic is transformed into the cross-sectional rank
    for i in cols:
        data[i] = 2*data.groupby('date')[i].rank(pct=True) - 1 

    return data

data = loadData()

def getCoef(path, year):
    data = pd.read_pickle(path)
    df = pd.DataFrame()

    for i in range(len(data)):
        m = data[i]
        tmp = pd.Series(m.coef_)
        df = pd.concat([df, tmp], axis=1)

    rng = pd.date_range(start='1995-1', end='2022-1', freq='A').year
    df = df.transpose()
    df.index = rng

    df.columns = cols
    coef = (df.loc[df.index == int(year)]).values
    return coef


obj_month_list = [int(f"{year}{month:02d}") for year in range(1995, 2022) for month in range(1, 13)]

def load_data_helper(month, path):
    data_train = data[data["date"].astype(str).str[:6] == str(month)]
    year = str(month)[:4]
    coef = getCoef(path, year)
    Y_train = data_train["ret_exc_lead1m"].to_numpy()
    X_train = data_train[cols].to_numpy()

    duplicated_coef = np.tile(coef, (len(X_train), 1))
    preds = np.sum(duplicated_coef * X_train, axis=1)
    
    size_train = data_train["market_equity"].to_numpy()
    pred_true_combined = np.vstack((preds,Y_train,size_train)).T
    pred_true_combined = pd.DataFrame(pred_true_combined)
    pred_true_combined.columns = ["pred","true","size"]
    pred_true_combined['true_size'] = pred_true_combined[["true"]].multiply(pred_true_combined["size"], axis="index")
    pred_true_combined = np.array(pred_true_combined)
    return pred_true_combined
     

def calc_portfolio(path):
    rets_ls_vw = np.zeros(len(obj_month_list))
    rets_l_vw = np.zeros(len(obj_month_list))

    for i in tqdm(range(len(obj_month_list))):
        pred_true_combined = load_data_helper(obj_month_list[i], path)
        num_stocks = pred_true_combined.shape[0] // 10
        highest_indices = get_top_10_percent_indices(pred_true_combined[:,0])
        lowest_indices = get_bottom_10_percent_indices(pred_true_combined[:,0])

        #value weighted portfolio return
        #rets_ls_vw is the excess return of the value-weighted long-short portfolio
        #rets_l_vw is the excess return of the value-weighted long portfolio
        rets_ls_vw[i] = 0.5*(np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2]) - 
                            np.sum(pred_true_combined[lowest_indices,3])/np.sum(pred_true_combined[lowest_indices,2]))
        rets_l_vw[i] = np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2])

    # print("------------long-short----------")
    # print(rets_ls_vw)
    # print("------------long-----------")
    # print(rets_l_vw)

    return rets_ls_vw, rets_l_vw


eNetPath = 'outputs/elasticnet/models.pkl'
eNet_long_short, eNet_long = calc_portfolio(eNetPath)

with open('outputs/portfolio/long_short_eNet.pkl', 'wb') as f:
    pickle.dump(eNet_long_short, f)
with open('outputs/portfolio/long_eNet.pkl', 'wb') as f:
    pickle.dump(eNet_long, f)

