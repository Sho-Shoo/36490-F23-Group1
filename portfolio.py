import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from classes.data_loader import DataLoader

dataloader = DataLoader('data/usa_short.csv')


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

cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d",
            "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1",
            "at_be", "cash_at", "age", "z_score"]

def loadData():
    data = pd.read_csv('data/usa_short.csv')
    
    data["market_equity1"] = data["market_equity"]

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


obj_month_list = [int(f"{year}{month:02d}") for year in range(1995, 2017) for month in range(1, 13)]

def load_data_helper(month, path):
    data_train = data[data["date"].astype(str).str[:6] == str(month)]
    year = str(month)[:4]
    coef = getCoef(path, year)
    Y_train = data_train["ret_exc_lead1m"].to_numpy()
    X_train = data_train[cols].to_numpy()

    duplicated_coef = np.tile(coef, (len(X_train), 1))
    preds = np.sum(duplicated_coef * X_train, axis=1)
    
    size_train = data_train["market_equity1"].to_numpy()
    pred_true_combined = np.vstack((preds,Y_train,size_train)).T
    pred_true_combined = pd.DataFrame(pred_true_combined)
    pred_true_combined.columns = ["pred","true","size"]
    pred_true_combined['true_size'] = pred_true_combined[["true"]].multiply(pred_true_combined["size"], axis="index")
    pred_true_combined = np.array(pred_true_combined)
    return pred_true_combined


def load_predictions(month, path):
    data_train = dataloader.slice(month*100 + 1, (month+1)*100)
    Y_train = dataloader.get_y(data_train)

    with open(path, 'rb') as f:
        all_preds = pickle.load(f)
        preds = all_preds[obj_month_list.index(month)]

    size_train = data_train["market_equity"].to_numpy()
    pred_true_combined = np.vstack((preds, Y_train, size_train)).T
    pred_true_combined = pd.DataFrame(pred_true_combined)
    pred_true_combined.columns = ["pred", "true", "size"]
    pred_true_combined['true_size'] = pred_true_combined[["true"]].multiply(pred_true_combined["size"], axis="index")
    pred_true_combined = np.array(pred_true_combined)
    return pred_true_combined
     

def calc_portfolio(path):
    rets_ls_vw = np.zeros(len(obj_month_list))
    rets_l_vw = np.zeros(len(obj_month_list))

    #for i in tqdm(range(len(obj_month_list))):
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


def calc_portfolio_from_predictions(path):
    rets_ls_vw = np.zeros(len(obj_month_list))
    rets_l_vw = np.zeros(len(obj_month_list))

    # for i in tqdm(range(len(obj_month_list))):
    for i in tqdm(range(len(obj_month_list))):
        pred_true_combined = load_predictions(obj_month_list[i], path)
        num_stocks = pred_true_combined.shape[0] // 10
        highest_indices = get_top_10_percent_indices(pred_true_combined[:, 0])
        lowest_indices = get_bottom_10_percent_indices(pred_true_combined[:, 0])

        #value weighted portfolio return
        #rets_ls_vw is the excess return of the value-weighted long-short portfolio
        #rets_l_vw is the excess return of the value-weighted long portfolio
        rets_ls_vw[i] = 0.5*(np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2]) -
                             np.sum(pred_true_combined[lowest_indices,3])/np.sum(pred_true_combined[lowest_indices,2]))
        rets_l_vw[i] = np.sum(pred_true_combined[highest_indices,3])/np.sum(pred_true_combined[highest_indices,2])

    return rets_ls_vw, rets_l_vw


if __name__ == "__main__":
    # eNetPredictions = 'outputs/elasticnet/predictions.pkl'
    # eNet_long_short, eNet_long = calc_portfolio_from_predictions(eNetPredictions)
    #
    # with open('outputs/portfolio/long_short_eNet.pkl', 'wb') as f:
    #     pickle.dump(eNet_long_short, f)
    # with open('outputs/portfolio/long_eNet.pkl', 'wb') as f:
    #     pickle.dump(eNet_long, f)

    # nn3Path = 'outputs/nn3/predictions.pkl'
    # nn3_long_short, nn3_long = calc_portfolio_from_predictions(nn3Path)
    #
    # with open('outputs/portfolio/long_short_nn3.pkl', 'wb') as f:
    #     pickle.dump(nn3_long_short, f)
    # with open('outputs/portfolio/long_nn3.pkl', 'wb') as f:
    #     pickle.dump(nn3_long, f)

    nn5Path = 'outputs/nn5_100epochs/predictions.pkl'
    nn5_long_short, nn5_long = calc_portfolio_from_predictions(nn5Path)

    with open('outputs/portfolio/long_short_nn5.pkl', 'wb') as f:
        pickle.dump(nn5_long_short, f)
    with open('outputs/portfolio/long_nn5.pkl', 'wb') as f:
        pickle.dump(nn5_long, f)
