"""
Input files:
1. new_merchant_transactions.csv (source: data/raw)
2. historical_transactions.csv (source: data/raw)
3. train.csv (source: data/check_point)
4. test.csv (source: data/check_point)

Output files:
1. /output/feature_correlation_main_lgb1.csv
2. /data/check_point/null_imp_df_1.feather
3. /data/check_point/actual_imp_df.csv
4. /final/data/check_point/test_c.csv
5. /final/data/check_point/train_c.csv
6. /data/check_point/test_checkpoint.feather
7. /data/check_point/train_checkpoint.feather
8. /data/check_point/new_transactions_checkpoint.feather
9. /data/check_point/historical_transactions_checkpoint.feather
"""

import datetime
import gc
import sys
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error,log_loss
from sklearn.model_selection import KFold, StratifiedKFold

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

new_transactions = reduce_mem_usage(pd.read_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/raw/new_merchant_transactions.csv', parse_dates=['purchase_date']))
historical_transactions = reduce_mem_usage(pd.read_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/raw/historical_transactions.csv', parse_dates=['purchase_date']))

new_transactions_copy = new_transactions.copy(deep=True)
historical_transactions_copy = historical_transactions.copy(deep=True)

# new_transactions = new_transactions_copy.copy(deep=True)
# historical_transactions = historical_transactions_copy.copy(deep=True)

print(new_transactions.columns.tolist())
print()
print(historical_transactions.columns.tolist())

# TODO: what is going on? 
new_transactions['purchase_amount'] = (new_transactions['purchase_amount']+0.761920783094309)*66.5423961054881
historical_transactions['purchase_amount'] = (historical_transactions['purchase_amount']+0.761920783094309)*66.5423961054881

def binarize(df):
    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})
    return df

historical_transactions = binarize(historical_transactions)
new_transactions = binarize(new_transactions)

# fill with 1.0 and A bcs its the top word
historical_transactions['category_2'].fillna(1.0,inplace=True)
historical_transactions['category_3'].fillna('A',inplace=True)

new_transactions['category_2'].fillna(1.0,inplace=True)
new_transactions['category_3'].fillna('A',inplace=True)


historical_transactions['category_3'] = historical_transactions['category_3'].map({'A':0, 'B':1, 'C':2})
new_transactions['category_3'] = new_transactions['category_3'].map({'A':0, 'B':1, 'C':2})

historical_transactions['purchase_date'] = pd.to_datetime(historical_transactions['purchase_date'])
new_transactions['purchase_date'] = pd.to_datetime(new_transactions['purchase_date'])

historical_transactions['year'] = historical_transactions['purchase_date'].dt.year
historical_transactions['weekofyear'] = historical_transactions['purchase_date'].dt.isocalendar().week
historical_transactions['month'] = historical_transactions['purchase_date'].dt.month
historical_transactions['dayofweek'] = historical_transactions['purchase_date'].dt.dayofweek
historical_transactions['weekend'] = (historical_transactions.purchase_date.dt.weekday >=5).astype(int)
historical_transactions['hour'] = historical_transactions['purchase_date'].dt.hour 
historical_transactions['quarter'] = historical_transactions['purchase_date'].dt.quarter
historical_transactions['is_month_start'] = historical_transactions['purchase_date'].dt.is_month_start

historical_transactions['month_diff'] = ((datetime.datetime.today() - historical_transactions['purchase_date']).dt.days)//30
historical_transactions['month_diff'] += historical_transactions['month_lag']
historical_transactions['category_2'] = historical_transactions['category_2'].astype('float32')

agg_func = {
        'mean': ['mean'],
    }

for col in ['category_2','category_3']:
    historical_transactions[col+'_mean'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('mean')
    historical_transactions[col+'_max'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('max')
    historical_transactions[col+'_min'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('min')
    historical_transactions[col+'_var'] = historical_transactions['purchase_amount'].groupby(historical_transactions[col]).agg('var')
    agg_func[col+'_mean'] = ['mean']

new_transactions['year'] = new_transactions['purchase_date'].dt.year
new_transactions['weekofyear'] = new_transactions['purchase_date'].dt.isocalendar().week
new_transactions['month'] = new_transactions['purchase_date'].dt.month
new_transactions['dayofweek'] = new_transactions['purchase_date'].dt.dayofweek
new_transactions['weekend'] = (new_transactions.purchase_date.dt.weekday >=5).astype(int)
new_transactions['hour'] = new_transactions['purchase_date'].dt.hour 
new_transactions['quarter'] = new_transactions['purchase_date'].dt.quarter
new_transactions['is_month_start'] = new_transactions['purchase_date'].dt.is_month_start

new_transactions['month_diff'] = ((datetime.datetime.today() - new_transactions['purchase_date']).dt.days)//30
new_transactions['month_diff'] += new_transactions['month_lag']
new_transactions['category_2'] = new_transactions['category_2'].astype('float32')

agg_func = {
        'mean': ['mean'],
    }

for col in ['category_2','category_3']:
    new_transactions[col+'_mean'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('mean')
    new_transactions[col+'_max'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('max')
    new_transactions[col+'_min'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('min')
    new_transactions[col+'_var'] = new_transactions['purchase_amount'].groupby(new_transactions[col]).agg('var')
    agg_func[col+'_mean'] = ['mean']

gc.collect()

# New Features with Key Shopping times considered in the dataset. if the purchase has been made within 60 days, it is considered as an influence
#Christmas : December 25 2017
historical_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#fathers day: August 13 2017
historical_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Childrens day: October 12 2017
historical_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Black Friday : 24th November 2017
historical_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Mothers Day: May 14 2017
historical_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-14') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-14') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
#Valentines Day
historical_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - historical_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
new_transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - new_transactions['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)


gc.collect()

# Calculating Mode
mode_list = historical_transactions.groupby('card_id')[['city_id', 'merchant_category_id', 'state_id', 'subsector_id', 'month_lag']].apply(lambda x: x.mode().iloc[0])
mode_list.columns = ['mode_' + c if c != 'card_id' else c for c in mode_list.columns]
# print(mode_list.columns.tolist())
historical_transactions = pd.merge(historical_transactions,mode_list,on='card_id',how='left')
# print(historical_transactions.columns.tolist())

# print()
mode_list = new_transactions.groupby('card_id')[['city_id','merchant_category_id','state_id','subsector_id','month_lag']].apply(lambda x: x.mode().iloc[0])
mode_list.columns = ['mode_' + c if c != 'card_id' else c for c in mode_list.columns]
# print(mode_list.columns.tolist())
new_transactions = pd.merge(new_transactions,mode_list,on='card_id',how='left')
# print(new_transactions.columns.tolist())

del mode_list;gc.collect()

# TODO: why the repitition? 
historical_transactions['purchase_month'] = historical_transactions['purchase_date'].dt.month
new_transactions['purchase_month'] = new_transactions['purchase_date'].dt.month

# print(new_transactions.columns.tolist())
# print()
# print(historical_transactions.columns.tolist())

historical_transactions.to_feather('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/historical_transactions_checkpoint.feather')
new_transactions.to_feather('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/new_transactions_checkpoint.feather')

# historical_transactions = pd.read_feather('checkpoint/historical_transactions_checkpoint.feather')
# new_transactions = pd.read_feather('checkpoint/new_transactions_checkpoint.feather')

agg_fun = {'authorized_flag': ['mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['auth_mean_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)
print(auth_mean.columns.tolist())

authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
print(authorized_transactions.columns.tolist())
non_authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]
print(non_authorized_transactions.columns.tolist())

hist_weekend_transactions = historical_transactions[historical_transactions['weekend'] == 1]
new_weekend_transactions = new_transactions[new_transactions['weekend'] == 1]
print(new_weekend_transactions.columns.tolist())

def read_data(input_file):
    df = pd.read_csv(input_file)

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    reference_date = pd.Timestamp('2018-02-01')
    df['elapsed_time'] = (reference_date - pd.to_datetime(df['first_active_month'])).dt.days
    return df

train = reduce_mem_usage(read_data('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/raw/train.csv'))
test = reduce_mem_usage(read_data('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/raw/test.csv'))

# print(train.columns.tolist())
# print(test.columns.tolist())

def aggregate_transactions(history):
    
    
    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': ['min', 'max'],
        'month_lag': ['min', 'max','std','var'],
        #
        'month_diff' : ['mean', 'min', 'max', 'var'],
        'weekend' : ['sum', 'mean'],
        'card_id' : ['size'],
        'month': ['nunique'],
        'hour': ['nunique'],
        'quarter':['nunique'],
        'weekofyear': ['nunique'],                
        'dayofweek': ['nunique'],
        'Christmas_Day_2017':['mean','max'],
        'fathers_day_2017':['mean','max'],
        'Children_day_2017':['mean','max'],        
        'Black_Friday_2017':['mean','max'],
        'Valentine_day_2017':['mean'],
        'Mothers_Day_2017':['mean'],
        #
        'mode_city_id':['mean'],
        'mode_merchant_category_id':['mean'],
        'mode_state_id':['mean'],
        'mode_subsector_id':['mean'],
        'mode_month_lag':['mean'],
    }

    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

print("Merge 1")
train = pd.merge(train, auth_mean, on='card_id', how='left')
test = pd.merge(test, auth_mean, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del auth_mean;gc.collect()

print("Merge 2")
history = aggregate_transactions(non_authorized_transactions)
history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
print(history.columns.tolist())
print()
del non_authorized_transactions;gc.collect()

train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del history;gc.collect()

print("Merge 3")
authorized = aggregate_transactions(authorized_transactions)
authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
print(authorized.columns.tolist())
print()
del authorized_transactions;gc.collect()

train = pd.merge(train, authorized, on='card_id', how='left')
test = pd.merge(test, authorized, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del authorized;gc.collect()

print("Merge 4")
new = aggregate_transactions(new_transactions)
new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
print(new.columns.tolist())

train = pd.merge(train, new, on='card_id', how='left')
test = pd.merge(test, new, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del new;gc.collect()

def aggregate_transactions_weekend(history):

    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        #
        'mode_city_id':['mean'],
        'mode_merchant_category_id':['mean'],
        'mode_state_id':['mean'],
        'mode_subsector_id':['mean'],
        'mode_month_lag':['mean'],
    }

    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

gc.collect()

print("Merge 5")
hist_weekend =aggregate_transactions_weekend(hist_weekend_transactions)
hist_weekend.columns = ['hist_weekend_' + c if c != 'card_id' else c for c in hist_weekend.columns]
print(hist_weekend.columns.tolist())
print()
del hist_weekend_transactions;gc.collect()

train = pd.merge(train, hist_weekend, on='card_id', how='left')
test = pd.merge(test, hist_weekend, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del hist_weekend;gc.collect()

print("Merge 6")
new_weekend =aggregate_transactions_weekend(new_weekend_transactions)
new_weekend.columns = ['new_weekend_' + c if c != 'card_id' else c for c in new_weekend.columns]
print(new_weekend.columns.tolist())
del new_weekend_transactions;gc.collect()

train = pd.merge(train, new_weekend, on='card_id', how='left')
test = pd.merge(test, new_weekend, on='card_id', how='left')
print(train.columns.tolist())
print(test.columns.tolist())
print()
del new_weekend;gc.collect()

print('processing hist merchant id')
#--------unique merchant_id  ------------------hist
m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_57df19bf28'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_57df19bf28'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_19171c737a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_19171c737a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_8fadd601d2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_8fadd601d2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9e84cda3b1'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9e84cda3b1'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_b794b9d9e8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_b794b9d9e8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_ec24d672a3'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_ec24d672a3'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5a0a412718'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5a0a412718'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_490f186c5a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_490f186c5a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_77e2942cd8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_77e2942cd8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_fc7d7969c3'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_fc7d7969c3'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_0a00fa9e8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_0a00fa9e8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1f4773aa76'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1f4773aa76'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5ba019a379'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5ba019a379'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_ae9fe1605a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_ae9fe1605a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_48257bb851'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_48257bb851'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1d8085cf5d'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1d8085cf5d'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_820c7b73c8'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_820c7b73c8'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_1ceca881f0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_1ceca881f0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_00a6ca8a8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_00a6ca8a8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_e5374dabc0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_e5374dabc0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9139332ccc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'hist_M_ID_9139332ccc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_5d4027918d'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_5d4027918d'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9fa00da7b2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9fa00da7b2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_d7f0a89a87'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_d7f0a89a87'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_daeb0fe461'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_daeb0fe461'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_077bbb4469'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_077bbb4469'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_f28259cb0a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_f28259cb0a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_0a767b8200'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_0a767b8200'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_fee47269cb'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_fee47269cb'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_c240e33141'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_c240e33141'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_a39e6f1119'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_a39e6f1119'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_c8911208f2'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_c8911208f2'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9a06a8cf31'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9a06a8cf31'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_08fdba20dc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_08fdba20dc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_a483a17d19'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_a483a17d19'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_aed77085ce'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_aed77085ce'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_25d0d2501a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_25d0d2501a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_69f024d01a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_69f024d01a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

del historical_transactions

print('processing new merchant id')
m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_6f274b9340'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_6f274b9340'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_445742726b'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_445742726b'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_4e461f7e14'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_4e461f7e14'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_e5374dabc0'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_e5374dabc0'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_3111c6df35'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_3111c6df35'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_50f575c681'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_50f575c681'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_00a6ca8a8a'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_00a6ca8a8a'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_cd2c0b07e9'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_cd2c0b07e9'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')

m = new_transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x=='M_ID_9139332ccc'].count())
m = m.to_frame()
m = m.rename(columns={'merchant_id':'M_ID_9139332ccc'})
train = pd.merge(train,m,on='card_id',how='left')
test = pd.merge(test,m,on='card_id',how='left')


del m, new_transactions;gc.collect()

train.to_feather('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/train_checkpoint.feather')
test.to_feather('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/test_checkpoint.feather')

#train = pd.read_feather('checkpoint/train_checkpoint.feather')
#test = pd.read_feather('checkpoint/test_checkpoint.feather')

train['rounded_target'] = train['target'].round(0)
train = train.sort_values('rounded_target').reset_index(drop=True)
vc = train['rounded_target'].value_counts()
vc = dict(sorted(vc.items()))
print(vc)

df = pd.DataFrame()
train['indexcol'],i = 0,1
for k,v in vc.items():
    step = train.shape[0]/v
    indent = train.shape[0]/(v+1)
    df2 = train[train['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
    for j in range(0, v):
        df2.at[j, 'indexcol'] = indent + j*step + 0.000001*i
    df = pd.concat([df2,df])
    i+=1
train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)

del train['indexcol'], train['rounded_target']

train['outliers'] = 0
train.loc[train['target'] < -30, 'outliers'] = 1

outliers = train['outliers']
target = train['target']
#del train['target']
gc.collect()

train["month"] = train["first_active_month"].dt.month
train["year"] = train["first_active_month"].dt.year
train['week'] = train["first_active_month"].dt.isocalendar().week
train['dayofweek'] = train['first_active_month'].dt.dayofweek
train['first_active_month'] = pd.to_datetime(train["first_active_month"])
train['days'] = (pd.Timestamp(2018, 2, 1) - train['first_active_month']).dt.days

test["month"] = test["first_active_month"].dt.month
test["year"] = test["first_active_month"].dt.year
test['week'] = test["first_active_month"].dt.isocalendar().week
test['dayofweek'] = test['first_active_month'].dt.dayofweek
train['first_active_month'] = pd.to_datetime(train["first_active_month"])
test['days'] = (pd.Timestamp(2018, 2, 1) - test['first_active_month']).dt.days

train['hist_purchase_date_max'] = pd.to_datetime(train['hist_purchase_date_max'])
train['hist_purchase_date_min'] = pd.to_datetime(train['hist_purchase_date_min'])
train['hist_purchase_date_diff'] = (train['hist_purchase_date_max'] - train['hist_purchase_date_min']).dt.days
train['hist_purchase_date_average'] = train['hist_purchase_date_diff']/train['hist_card_id_size']
train['hist_purchase_date_uptonow'] = (datetime.datetime.today() - train['hist_purchase_date_max']).dt.days
train['hist_purchase_date_uptomin'] = (datetime.datetime.today() - train['hist_purchase_date_min']).dt.days
train['hist_first_buy'] = (train['hist_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['hist_purchase_date_max'] = pd.to_datetime(test['hist_purchase_date_max'])
test['hist_purchase_date_min'] = pd.to_datetime(test['hist_purchase_date_min'])
test['hist_purchase_date_diff'] = (test['hist_purchase_date_max'] - test['hist_purchase_date_min']).dt.days
test['hist_purchase_date_average'] = test['hist_purchase_date_diff']/test['hist_card_id_size']
test['hist_purchase_date_uptonow'] = (datetime.datetime.today() - test['hist_purchase_date_max']).dt.days
test['hist_purchase_date_uptomin'] = (datetime.datetime.today() - test['hist_purchase_date_min']).dt.days
test['hist_first_buy'] = (test['hist_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['hist_purchase_date_max','hist_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

# print(train.columns.tolist())
# print(test.columns.tolist())

train['auth_purchase_date_max'] = pd.to_datetime(train['auth_purchase_date_max'])
train['auth_purchase_date_min'] = pd.to_datetime(train['auth_purchase_date_min'])
train['auth_purchase_date_diff'] = (train['auth_purchase_date_max'] - train['auth_purchase_date_min']).dt.days
train['auth_purchase_date_average'] = train['auth_purchase_date_diff']/train['auth_card_id_size']
train['auth_purchase_date_uptonow'] = (datetime.datetime.today() - train['auth_purchase_date_max']).dt.days
train['auth_purchase_date_uptomin'] = (datetime.datetime.today() - train['auth_purchase_date_min']).dt.days
train['auth_first_buy'] = (train['auth_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['auth_purchase_date_max','auth_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['auth_purchase_date_max'] = pd.to_datetime(test['auth_purchase_date_max'])
test['auth_purchase_date_min'] = pd.to_datetime(test['auth_purchase_date_min'])
test['auth_purchase_date_diff'] = (test['auth_purchase_date_max'] - test['auth_purchase_date_min']).dt.days
test['auth_purchase_date_average'] = test['auth_purchase_date_diff']/test['auth_card_id_size']
test['auth_purchase_date_uptonow'] = (datetime.datetime.today() - test['auth_purchase_date_max']).dt.days
test['auth_purchase_date_uptomin'] = (datetime.datetime.today() - test['auth_purchase_date_min']).dt.days
test['auth_first_buy'] = (test['auth_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['auth_purchase_date_max','auth_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

# print(train.columns.tolist())
# print(test.columns.tolist())

train['new_purchase_date_max'] = pd.to_datetime(train['new_purchase_date_max'])
train['new_purchase_date_min'] = pd.to_datetime(train['new_purchase_date_min'])
train['new_purchase_date_diff'] = (train['new_purchase_date_max'] - train['new_purchase_date_min']).dt.days
train['new_purchase_date_average'] = train['new_purchase_date_diff']/train['new_card_id_size']
train['new_purchase_date_uptonow'] = (datetime.datetime.today() - train['new_purchase_date_max']).dt.days
train['new_purchase_date_uptomin'] = (datetime.datetime.today() - train['new_purchase_date_min']).dt.days
train['new_first_buy'] = (train['new_purchase_date_min'] - train['first_active_month']).dt.days
for feature in ['new_purchase_date_max','new_purchase_date_min']:
    train[feature] = train[feature].astype(np.int64) * 1e-9

test['new_purchase_date_max'] = pd.to_datetime(test['new_purchase_date_max'])
test['new_purchase_date_min'] = pd.to_datetime(test['new_purchase_date_min'])
test['new_purchase_date_diff'] = (test['new_purchase_date_max'] - test['new_purchase_date_min']).dt.days
test['new_purchase_date_average'] = test['new_purchase_date_diff']/test['new_card_id_size']
test['new_purchase_date_uptonow'] = (datetime.datetime.today() - test['new_purchase_date_max']).dt.days
test['new_purchase_date_uptomin'] = (datetime.datetime.today() - test['new_purchase_date_min']).dt.days
test['new_first_buy'] = (test['new_purchase_date_min'] - test['first_active_month']).dt.days
for feature in ['new_purchase_date_max','new_purchase_date_min']:
    test[feature] = test[feature].astype(np.int64) * 1e-9

# print(train.columns.tolist())
# print(test.columns.tolist())

train['diff_purchase_date_diff'] =  train['auth_purchase_date_diff'] - train['new_purchase_date_diff'] 
train['diff_purchase_date_average'] = train['auth_purchase_date_average'] - train['new_purchase_date_average']
train['hist_00a6ca8a8a_ratio'] = train['hist_M_ID_00a6ca8a8a']/(train['hist_transactions_count']+train['auth_transactions_count'])
train['new_00a6ca8a8a_ratio'] = train['M_ID_00a6ca8a8a']/train['new_transactions_count']

test['diff_purchase_date_diff'] =  test['auth_purchase_date_diff'] - test['new_purchase_date_diff']
test['diff_purchase_date_average'] = test['auth_purchase_date_average'] - test['new_purchase_date_average']
test['hist_00a6ca8a8a_ratio'] = test['hist_M_ID_00a6ca8a8a']/(test['hist_transactions_count']+test['auth_transactions_count'])
test['new_00a6ca8a8a_ratio'] = test['M_ID_00a6ca8a8a']/test['new_transactions_count']


# print(train.columns.tolist())
# print(test.columns.tolist())

train['category1_auth_ratio'] = train['auth_category_1_sum']/train['auth_transactions_count']
train['category_1_new_ratio'] = train['new_category_1_sum']/train['new_transactions_count']
train['date_average_new_auth_ratio'] = train['auth_purchase_date_average']/train['new_purchase_date_average']
train['childday_ratio'] = train['auth_Children_day_2017_mean']/train['new_Children_day_2017_mean']
train['blackday_ratio'] = train['auth_Black_Friday_2017_mean']/train['new_Black_Friday_2017_mean']
train['fatherday_ratio'] = train['auth_fathers_day_2017_mean']/train['new_fathers_day_2017_mean']
train['christmasday_ratio'] = train['auth_Christmas_Day_2017_mean']/train['new_Christmas_Day_2017_mean']
train['date_uptonow_diff_auth_new'] = train['auth_purchase_date_uptonow'] - train['new_purchase_date_uptonow']

test['category1_auth_ratio'] = test['auth_category_1_sum']/test['auth_transactions_count']
test['category_1_new_ratio'] = test['new_category_1_sum']/test['new_transactions_count']
test['date_average_new_auth_ratio'] = test['auth_purchase_date_average']/test['new_purchase_date_average']
test['childday_ratio'] = test['auth_Children_day_2017_mean']/test['new_Children_day_2017_mean']
test['blackday_ratio'] = test['auth_Black_Friday_2017_mean']/test['new_Black_Friday_2017_mean']
test['fatherday_ratio'] = test['auth_fathers_day_2017_mean']/test['new_fathers_day_2017_mean']
test['christmasday_ratio'] = test['auth_Christmas_Day_2017_mean']/test['new_Christmas_Day_2017_mean']
test['date_uptonow_diff_auth_new'] = test['auth_purchase_date_uptonow'] - test['new_purchase_date_uptonow']

# print(train.columns.tolist())
# print(test.columns.tolist())

train['category1_hist_weekend_ratio'] = train['hist_weekend_category_1_sum']/train['hist_weekend_transactions_count']
train['category_1_new_weekend_ratio'] = train['new_weekend_category_1_sum']/train['new_weekend_transactions_count']
test['category1_hist_weekend_ratio'] = test['hist_weekend_category_1_sum']/test['hist_weekend_transactions_count']
test['category_1_new_weekend_ratio'] = test['new_weekend_category_1_sum']/test['new_weekend_transactions_count']

# print(train.columns.tolist())
# print(test.columns.tolist())

train = train.fillna(0)
test = test.fillna(0)

train.to_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/train_c.csv')
test.to_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/test_c.csv')

features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
print(f"features: {features}")
print(f"categorical_feats: {categorical_feats}")

train = train[features]
test = test[features]

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
gc.collect()

def get_feature_importances(train, target, shuffle, seed=None):
    y = target
    if shuffle:
        y = pd.DataFrame(y).sample(frac=1.0)

    train['week'] = train['week'].astype('int32')

    # Pass categorical features correctly here
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, categorical_feature=categorical_feats)

    lgb_params = {
    'device': 'gpu',  # <- USE GPU
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_leaves': 31,
    'min_data_in_leaf': 32,
    'objective': 'regression',
    'max_depth': -1,
    'learning_rate': 0.005,
    'min_child_samples': 20,
    'boosting': 'gbdt',
    'feature_fraction': 0.9,
    'bagging_freq': 1,
    'bagging_fraction': 0.9,
    'bagging_seed': 11,
    'metric': 'rmse',
    'lambda_l1': 0.1,
    'verbose': -1
    }
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=3500)

    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))

    return imp_df

actual_imp_df = get_feature_importances(train=train,target=target,shuffle=False)

actual_imp_df.to_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/actual_imp_df.csv')

actual_imp_df.head(2)

actual_imp_df_sorted = actual_imp_df.sort_values(by='importance_gain', ascending=False)

plt.figure(figsize=(10, 55))
plt.barh(actual_imp_df_sorted['feature'], actual_imp_df_sorted['importance_gain'], color='skyblue')
plt.xlabel('Importance Gain')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()

null_imp_df = pd.DataFrame()
nb_runs = 80
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(train=train,target=target,shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='')


null_imp_df.to_feather("/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/null_imp_df_1.feather")

null_imp_df = pd.read_feather("/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/data/check_point/null_imp_df_1.feather")

def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())

display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='auth_first_buy')

display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='category1_hist_weekend_ratio')

display_distributions(actual_imp_df_=actual_imp_df, null_imp_df_=null_imp_df, feature_='new_month_nunique')

correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values  # get all the importance value of that particular feature from all runs
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values # get the actual feature importance    
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size

    correlation_scores.append((_f, split_score, gain_score))

print(len(correlation_scores))

scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
print(type(scores_df.loc[0, 'split_score']))  # Access by row index and column name
scores_df

plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2)
# Plot Split importances
ax = plt.subplot(gs[0, 0])
sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).head(30), ax=ax)
ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
# Plot Gain importances
ax = plt.subplot(gs[0, 1])
sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).head(30), ax=ax)
ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
plt.tight_layout()

corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
gc.collect()

corr_scores_df.to_csv('/home/UG/aarushi003/joanne_solution/Elo-Merchant-Competition/final/output/feature_correlation_main_lgb1.csv', index=False)
