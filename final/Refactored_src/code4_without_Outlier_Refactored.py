"""
Purpose:
Compute feature importance scores using LightGBM **after filtering out outliers** from the training set (i.e., `outliers == 0`).
This isolates the influence of features under normal data distribution and reduces noise from extreme values.
Also uses null importance analysis with shuffled targets to calculate gain/split scores.

Input Files:
1. /home/UG/pareena001/SC4000-ML-Grp1/data/train_c.csv
2. /home/UG/pareena001/SC4000-ML-Grp1/data/test_c.csv

Output Files:
1. /home/UG/pareena001/SC4000-ML-Grp1/data/check_point/correlation_without_outlier.feather
2. /home/UG/pareena001/SC4000-ML-Grp1/data/processed/feature_correlation_without_outlier.csv
"""

import time
import gc
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Suppress warnings
def suppress_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', RuntimeWarning)

# File paths
TRAIN_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/train_c.csv'
TEST_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/test_c.csv'
NULL_IMP_FEATHER_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/check_point/correlation_without_outlier.feather'
CORR_CSV_PATH = '/home/UG/pareena001/SC4000-ML-Grp1/data/processed/feature_correlation_without_outlier.csv'

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

def load_and_preprocess_data(train_path, test_path):
    train = reduce_mem_usage(pd.read_csv(train_path))
    test = reduce_mem_usage(pd.read_csv(test_path))
    if 'Unnamed: 0' in train.columns:
        train.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in test.columns:
        test.drop('Unnamed: 0', axis=1, inplace=True)

    train = train[train['outliers'] == 0]
    target = train['target']

    excluded = ['card_id', 'target', 'first_active_month', 'outliers',
                'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
                'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max']
    features = [c for c in train.columns if c not in excluded]
    categorical_feats = [c for c in features if 'feature_' in c]

    train = train[features]
    test = test[features]

    if 'week' in train.columns:
        train['week'] = train['week'].astype('int32')

    return train, test, target, features, categorical_feats

def get_feature_importances(train, target, features, categorical_feats, shuffle=False, seed=None):
    y = target.sample(frac=1.0, random_state=seed).reset_index(drop=True) if shuffle else target
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, categorical_feature=categorical_feats)

    params = {
        'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0,
        'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse',
        'learning_rate': 0.005, 'num_leaves': 31, 'min_data_in_leaf': 32,
        'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1,
        'lambda_l1': 0.1, 'bagging_seed': 11, 'verbosity': -1, 'nthread': 8
    }

    model = lgb.train(params, dtrain, num_boost_round=3500)

    return pd.DataFrame({
        'feature': features,
        'importance_gain': model.feature_importance(importance_type='gain'),
        'importance_split': model.feature_importance(importance_type='split'),
        'trn_score': mean_squared_error(y, model.predict(train[features]))
    })

def run_null_importance(train, target, features, categorical_feats, nb_runs=80):
    actual_imp_df = get_feature_importances(train, target, features, categorical_feats, shuffle=False)
    null_imp_df = pd.DataFrame()
    start = time.time()

    for i in range(nb_runs):
        imp_df = get_feature_importances(train, target, features, categorical_feats, shuffle=True, seed=i)
        imp_df['run'] = i + 1
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        print(f"\rRun {i+1}/{nb_runs} complete. Elapsed: {(time.time()-start)/60:.1f} min", end='')

    print()
    null_imp_df.reset_index(drop=True, inplace=True)
    return actual_imp_df, null_imp_df

def calculate_correlation_scores(actual_imp_df, null_imp_df):
    correlation_scores = []
    for feat in actual_imp_df['feature'].unique():
        null_gain = null_imp_df.loc[null_imp_df['feature'] == feat, 'importance_gain'].values
        actual_gain = actual_imp_df.loc[actual_imp_df['feature'] == feat, 'importance_gain'].values
        gain_score = 100 * (null_gain < np.percentile(actual_gain, 25)).sum() / null_gain.size

        null_split = null_imp_df.loc[null_imp_df['feature'] == feat, 'importance_split'].values
        actual_split = actual_imp_df.loc[actual_imp_df['feature'] == feat, 'importance_split'].values
        split_score = 100 * (null_split < np.percentile(actual_split, 25)).sum() / null_split.size

        correlation_scores.append((feat, split_score, gain_score))

    return pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

def main():
    # Suppress warnings
    suppress_warnings()

    train, test, target, features, categorical_feats = load_and_preprocess_data(TRAIN_PATH, TEST_PATH)
    print("[INFO] Starting null importance calculation with 80 runs")
    actual_imp_df, null_imp_df = run_null_importance(train, target, features, categorical_feats, nb_runs=80)

    print(f"[INFO] Saving null importances to: {NULL_IMP_FEATHER_PATH}")
    null_imp_df.to_feather(NULL_IMP_FEATHER_PATH)

    print("[INFO] Calculating correlation scores")
    corr_scores_df = calculate_correlation_scores(actual_imp_df, null_imp_df)
    print(f"[INFO] Analyzed {len(corr_scores_df)} features")

    print(f"[INFO] Writing correlation scores to: {CORR_CSV_PATH}")
    corr_scores_df.to_csv(CORR_CSV_PATH, index=False)
    gc.collect()
    print("[SUCCESS] Feature importance analysis completed")

if __name__ == "__main__":
    main()