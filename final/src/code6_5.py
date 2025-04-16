"""
Input Files:
1. train_c.csv (source: code1.py)
2. test_c.csv (source: code1.py)
3. feature_correlation_without_outlier.csv (source: code4_withoutOutlier.py)
4. sample_submission.csv (source: data/raw)

Output Files:
1. bestline_submission_without_outliers.csv
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
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import precision_score

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

train = reduce_mem_usage(pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/train_c.csv'))
test = reduce_mem_usage(pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/test_c.csv'))

train = train[train['outliers'] ==0]
target = train['target']
features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
train = train[features]
test = test[features]

corr_scores_df = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/processed/feature_correlation_without_outlier.csv')

threshold = 70
featuresFM = []
categorical_feats_split = []
for _f in corr_scores_df.itertuples():
    if _f[2] >= threshold:
        featuresFM.append(_f[1])

for  _f in corr_scores_df.itertuples():
    if (_f[2] >= threshold) & (_f[1] in categorical_feats):
        categorical_feats_split.append(_f[1])

print(len(featuresFM))
train = train[featuresFM]
test = test[featuresFM]


features = train.columns.values
categorical_feats = [c for c in features if 'feature_' in c]

param = {
    'device':'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'num_leaves': 31,
    'min_data_in_leaf': 32, 
    'objective':'regression',
    'max_depth': -1,
    'learning_rate': 0.005,
    "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.1,
         "nthread": 7,
         "verbosity": -1}

params ={'device':'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'subsample': 0.9855232997390695,
            'max_depth': 7,
            'top_rate': 0.9064148448434349,
            'num_leaves': 63,
            'min_child_weight': 41.9612869171337,
            'other_rate': 0.0721768246018207,
            'reg_alpha': 9.677537745007898,
            'colsample_bytree': 0.5665320670155495,
            'min_split_gain': 9.820197773625843,
            'reg_lambda': 8.2532317400459,
            'min_data_in_leaf': 21,
            'verbose': -1,
            'seed':int(2**7),
            'bagging_seed':int(2**7),
            'drop_seed':int(2**7)
            }

folds = KFold(n_splits=7, shuffle=False)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start = time.time()
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000

    # clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], early_stopping_rounds = 400)
    clf = lgb.train(param, trn_data, num_round,
                valid_sets=[trn_data, val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=400)])
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

sample_submission = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/raw/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_without_outliers.csv', index=False)



