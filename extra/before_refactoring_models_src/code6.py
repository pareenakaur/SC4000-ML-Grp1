"""
Input Files:
1. train_c.feather (source: code1.py)
2. test_c.feather (source: code1.py)
3. feature_correlation_only_outlier.csv (source: code2_withoutOutlier.py)
4. sample_submission.csv (source: /data/raw )

Output Files:
1. bestline_submission_outliers_likelihood2.csv

"""
from sklearn.metrics import f1_score
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

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

train = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/train_c.csv')
train['week'] = train['week'].astype('int32')
test = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/check_point/test_c.csv')
test['week' ]= test['week'].astype('int32')

target = train['outliers']

features = [c for c in train.columns if c not in ['card_id', 'target','first_active_month','outliers','hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max','new_weekend_purchase_date_min','new_weekend_purchase_date_max']]
categorical_feats = [c for c in features if 'feature_' in c]
print(f"features: {features}")
print(f"categorical_feats: {categorical_feats}")

train = train[features]
test = test[features]

corr_scores_df = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/processed/feature_correlation_only_outlier.csv')
threshold = 10
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

print(target.unique())

features = train.columns.values
categorical_feats = [c for c in features if 'feature_' in c]

param = {'device':'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.005,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'f1',
         "lambda_l1": 0.1,
         "nthread": 7,
         "verbosity": -1,
         "is_unbalance":'true'}

params ={'device':'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'binary',
            'metric': 'f1',
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
            'drop_seed':int(2**7),
            "is_unbalance":'true'
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

    # clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data],feval=lgb_f1_score ,verbose_eval=100, early_stopping_rounds = 400)
    clf = lgb.train(param, trn_data, num_round,
                valid_sets=[trn_data, val_data],
                feval=lgb_f1_score,
                callbacks=[lgb.early_stopping(stopping_rounds=400)])
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

sample_submission = pd.read_csv('/home/UG/aarushi003/SC4000_Final/data/raw/sample_submission.csv')
sample_submission['target'] = predictions
sample_submission.to_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_outliers_likelihood2.csv', index=False)

# model_without_outliers = pd.read_csv('./bestline_submission_without_outliers.csv')
# df_outlier_prob = pd.read_csv('checkpoint/bestline_submission_outliers_likelihood.csv')
# outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
# best_submission= pd.read_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/bestline_submission_main.csv')

# most_likely_liers = best_submission.merge(outlier_id,how='right')

# for card_id in most_likely_liers['card_id']:
#     model_without_outliers.loc[model_without_outliers["card_id"].isin(outlier_id["card_id"].values), "target"] = best_submission[best_submission["card_id"].isin(outlier_id["card_id"].values)]["target"]
# #%%
# model_without_outliers.to_csv('/home/takehiroo/デスクトップ/kaggle/Elo Merchant Category Recommendation/Final line/combining_submission.csv', index=False)

