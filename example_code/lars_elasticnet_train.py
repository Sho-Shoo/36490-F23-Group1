import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

def r2oos(y, yhat):
    num = np.sum((y- yhat)**2)
    den = np.sum((y)**2)
    return 1 - num/den
    

cols = ["be_me", "ret_12_1", "market_equity", "ret_1_0", "rvol_252d", "beta_252d", "qmj_safety", "rmax1_21d",
        "chcsho_12m", "ni_me", "eq_dur", "ret_60_12", "ope_be", "gp_at", "ebit_sale", "at_gr1", "sale_gr1",
        "at_be", "cash_at", "age", "z_score"]

data = pd.read_csv('../data/usa_short.csv')

data['date_str'] = data['date']
data['date'] = pd.to_datetime(data['date_str'], format='%Y%m%d') 
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

data = data[data['year']>=1980]

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


# Check for missing values
df = data
null = df[df.isnull().any(axis=1)]


alpha_grid = np.logspace(-10, -2, 100) # [0.0001, 0.001, ..., 10000]
l1_ratio_grid = np.linspace(0, 1, 21)  # [0.0, 0.1, ..., 1.0]

validation_r2s = []
test_r2s = []
alphas = []
l1_ratios = []
models = []



for train_start in tqdm(range(1980,2007)):
#for train_start in range (1980,1981):

    #t = time.time()
    train_end = train_start + 9
    validate_start = train_end + 1
    validate_end = validate_start + 4
    test_start = validate_end + 1
    test_end = test_start
    
    x_train = data[ (data['year']>=train_start) & (data['year']<=train_end) ]
    x_train = x_train.drop( x_train[(x_train['year']==train_end) & (x_train['month']==12)].index )
    y_train = x_train['ret_exc_lead1m'] 
    x_train = x_train[cols]
    
    x_validate = data[ (data['year']>=validate_start) & (data['year']<=validate_end) ]
    x_validate = x_validate.drop( x_validate[(x_validate['year']==validate_end) & (x_validate['month']==12)].index )

    y_validate = x_validate['ret_exc_lead1m'] 
    x_validate = x_validate[cols]
    
    best_r2, best_model, best_alpha, best_l1 = -1, None, None, None
    for alpha in alpha_grid:
        for l1 in l1_ratio_grid:
            print('alpha=',alpha,'L1=',l1)
            model = ElasticNet(alpha=alpha, l1_ratio=l1)
            model.fit(x_train, y_train)
            preds = model.predict(x_validate)
            r2 = r2oos(y_validate, preds)
            if r2 > best_r2:
                best_r2, best_model, best_alpha, best_l1 = r2, model, alpha, l1
    
    validation_r2s.append(best_r2)
    alphas.append(best_alpha)
    l1_ratios.append(best_l1)
    models.append(best_model)

    x_test = data[ (data['year']>=test_start) & (data['year']<=test_end) ]
    y_test = x_test[['ret_exc_lead1m','month']] 
    
    y_pred = best_model.predict(x_test[cols])
    y_pred = pd.DataFrame(y_pred,columns=['pred_ret'],index=x_test.index)
    y_pred = pd.concat([y_pred,x_test['month']],axis=1)

    # Calculate R-squared for the month
    for month in range(1, 13):
        y_test_month = y_test[y_test['month']==month]['ret_exc_lead1m']
        y_pred_month = y_pred[y_pred['month']==month]['pred_ret']
        
        r2 = r2oos(y_test_month, y_pred_month)
        test_r2s.append(r2)
    #elapsed = time.time() - t
    #print(elapsed)



with open('output/elasticnet/test_r2s.pkl', 'wb') as f:
    pickle.dump(test_r2s, f)

with open('output/elasticnet/validation_r2s.pkl', 'wb') as f:
    pickle.dump(validation_r2s, f)
    
with open('output/elasticnet/alphas.pkl', 'wb') as f:
    pickle.dump(alphas, f)

with open('output/elasticnet/l1_ratios.pkl', 'wb') as f:
    pickle.dump(l1_ratios, f)

with open('output/elasticnet/models.pkl', 'wb') as f:
    pickle.dump(models, f)

#with open('outputs/elasticnet/predictions.pkl', 'wb') as f:
#    pickle.dump(predictions, f)



import requests
message = 'The elasticnet code is done!'
webhook_url = 'https://discord.com/api/webhooks/1169268438506676297/D8qHeJPAAQn8Y0FvTUa3-DMNMBc1FEze_x4e2wem3vuHdZJM4Uyjv8zC2J3_VGwwch7R'
payload = {'content': message}  # Payload containing the message
response = requests.post(webhook_url, json=payload)  # Send POST request with JSON payload
    




'''
validation_r2s.to_pickle('outputs/elasticnet/validation_r2s.pkl')
test_r2s.to_pickle('outputs/elasticnet/test_r2s.pkl')
alphas.to_pickle('outputs/elasticnet/alphas.pkl')
l1_ratios.to_pickle('outputs/elasticnet/l1_ratios.pkl')
models.to_pickle('outputs/elasticnet/models.pkl')
monthly_r2_scores.to_pickle('outputs/elasticnet/test_r2s.pkl')
'''

