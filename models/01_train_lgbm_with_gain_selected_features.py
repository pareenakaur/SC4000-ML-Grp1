"""
Purpose:
Train a LightGBM model using features selected based on gain scores. Perform 7-fold cross-validation,
evaluate predictions with RMSE, and generate final output submission.

Input Files:
1. /home/UG/pareena001/SC4000-ML-Grp1/data/check_point/train_c.csv
2. /home/UG/pareena001/SC4000-ML-Grp1/data/check_point/test_c.csv
3. /home/UG/pareena001/SC4000-ML-Grp1/final/output/feature_correlation_main_lgb1.csv
4. /home/UG/pareena001/SC4000-ML-Grp1/data/sample_submission.csv

Output Files:
1. /home/UG/pareena001/SC4000-ML-Grp1/final/output/bestline_submission_main_2.csv
"""

import time
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# File paths
TRAIN_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/check_point/train_c.csv'
TEST_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/check_point/test_c.csv'
FEAT_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/final/output/feature_correlation_main_lgb1.csv'
SAMPLE_SUB_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/sample_submission.csv'
OUTPUT_SUB_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/final/output/bestline_submission_main_2.csv'

# Suppress warnings
def suppress_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', RuntimeWarning)

# Memory optimization
def reduce_mem_usage(df):
    for col in df.select_dtypes(include=['int', 'float']).columns:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(df[col].dtype).startswith('int'):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def load_data():
    train = reduce_mem_usage(pd.read_csv(TRAIN_PATH))
    test = reduce_mem_usage(pd.read_csv(TEST_PATH))
    feat_scores = pd.read_csv(FEAT_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    return train, test, feat_scores, sample_sub

def filter_features(train, test, feat_scores, threshold=70):
    exclude_cols = ['card_id', 'target','first_active_month','outliers',
                    'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
                    'new_weekend_purchase_date_min','new_weekend_purchase_date_max']
    features = [c for c in train.columns if c not in exclude_cols]
    categorical_feats = [c for c in features if 'feature_' in c]

    selected_feats = feat_scores[feat_scores['gain_score'] >= threshold]['feature'].tolist()
    selected_categoricals = [f for f in selected_feats if f in categorical_feats]

    train = train[selected_feats]
    test = test[selected_feats]
    return train, test, selected_feats, selected_categoricals

def train_model(train, test, target, features, categorical_feats):
    kf = KFold(n_splits=7, shuffle=False)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    params = {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'boosting': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.005,
        'num_leaves': 31,
        'min_data_in_leaf': 32,
        'min_child_samples': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_seed': 11,
        'lambda_l1': 0.1,
        'verbosity': -1,
        'nthread': 7
    }

    for fold, (trn_idx, val_idx) in enumerate(kf.split(train.values, target.values)):
        print(f"Fold {fold + 1}")
        X_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]
        X_val, y_val = train.iloc[val_idx], target.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_feats)
        dvalid = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_feats)

        model = lgb.train(params, dtrain, num_boost_round=10000,
                          valid_sets=[dtrain, dvalid],
                          callbacks=[lgb.early_stopping(stopping_rounds=400)])

        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        predictions += model.predict(test[features], num_iteration=model.best_iteration) / kf.n_splits

        fold_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importance(),
            'fold': fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

    print(f"CV RMSE: {mean_squared_error(target, oof) ** 0.5:.5f}")
    return predictions

def main():
    suppress_warnings()
    train, test, feat_scores, sample_sub = load_data()
    target = train['target']
    train, test, selected_feats, selected_categoricals = filter_features(train, test, feat_scores)
    predictions = train_model(train, test, target, selected_feats, selected_categoricals)

    sample_sub['target'] = predictions
    sample_sub.to_csv(OUTPUT_SUB_PATH, index=False)
    print("[SUCCESS] Submission file saved.")

if __name__ == "__main__":
    main()