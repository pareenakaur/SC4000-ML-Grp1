"""
Input files:
1. bestline_submission_without_outliers.csv (source: code6_5.py)
2. bestline_submission_outliers_likelihood2.csv (source: code6.py)
3. bestline_submission_main_2.csv (source: code3.py) 

Output files:
1. final_submission_threshold0.5.csv
2. final_submission_threshold0.05.csv
3. final_submission_threshold0.005.csv
4. final_submission.csv
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

from tqdm import tqdm


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)

threshold = [0.5, 0.05, 0.005]

for th in threshold:
    model_without_outliers = pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_without_outliers.csv')
    df_outlier_prob = pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_outliers_likelihood2.csv')
    outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False))
    outlier_id = outlier_id[outlier_id["target"] > th]

    best_submission= pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/code3_bestline_submission_main_2.csv')

    most_likely_liers = best_submission.merge(outlier_id,how='right')

    for card_id in tqdm(most_likely_liers['card_id'], desc="Updating targets", leave=False):
        model_without_outliers.loc[model_without_outliers["card_id"].isin(outlier_id["card_id"].values), "target"] = best_submission[best_submission["card_id"].isin(outlier_id["card_id"].values)]["target"]
    model_without_outliers.to_csv(f'/home/UG/aarushi003/SC4000_Final/data/processed/final_submission_threshold{th}.csv', index=False)

model_without_outliers = pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_without_outliers.csv')
df_outlier_prob = pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/bestline_submission_outliers_likelihood2.csv')
outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
best_submission= pd.read_csv('/home/UG/aarushi003/SC4000_Final/output/code3_bestline_submission_main_2.csv')

most_likely_liers = best_submission.merge(outlier_id,how='right')

for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers["card_id"].isin(outlier_id["card_id"].values), "target"] = best_submission[best_submission["card_id"].isin(outlier_id["card_id"].values)]["target"]
model_without_outliers.to_csv('/home/UG/aarushi003/SC4000_Final/output/final_submission.csv', index=False)




