"""
Input Files:
1. /final/data/check_point/train_checkpoint.feather (source: code1.py)
2. /final/data/check_point/test_checkpoint.feather (source: code1.py)

Output Files:
1. /data/check_point/null_imp_df_2.feather
2. feature_correlation_main_lgb2.csv

"""

import datetime
import gc
import sys
import time
import warnings

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error,log_loss
from sklearn.model_selection import KFold, StratifiedKFold

from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

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

train = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/train_c.csv')
test = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/test_c.csv')
target = train['target']

features = [c for c in train.columns if c not in ['Unnamed: 0', 'card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
print(f"features: {features}")
print(f"categorical_feats: {categorical_feats}")

train = train[features]
test = test[features]

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
gc.collect()

def get_feature_importances2(train,target, shuffle, seed=None):

    # Go over fold and keep track of CV score (train and valid) and feature importances
    train['week'] = train['week'].astype('int32')

    # Shuffle target if required
    y = target
    if shuffle:
        # Here you could as well use a binomial distribution
        y = pd.DataFrame(y).sample(frac=1.0)
    
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    # dtrain = lgb.Dataset(train[features], y, free_raw_data=False)
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, categorical_feature=categorical_feats)

    params ={
            'device':'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'task': 'train',
            'boosting': 'goss',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9,
            'max_depth': 7,
            'top_rate': 0.9,
            'num_leaves': 63,
            'min_child_weight': 40.,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.,
            'colsample_bytree': 0.5,
            'min_split_gain': 9,
            'reg_lambda': 8.,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**7),
            'bagging_seed':int(2**7),
            'drop_seed':int(2**7)
            }      
    
    
    # Fit the model
    # clf = lgb.train(params=params, train_set=dtrain, num_boost_round=3500, categorical_feature=categorical_feats)
    clf = lgb.train(params=params, train_set=dtrain, num_boost_round=3500)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))
    
    return imp_df

actual_imp_df = get_feature_importances2(train=train,target=target,shuffle=False)

null_imp_df = pd.DataFrame()
nb_runs = 80
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances2(train=train,target=target,shuffle=True)
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

null_imp_df.to_feather("/home/UG/aarushi003/SC4000_Final/data/check_point/null_imp_df_2.feather")

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

corr_scores_df.to_csv('/home/UG/aarushi003/SC4000_Final/data/processed/feature_correlation_main_lgb2.csv', index=False)

