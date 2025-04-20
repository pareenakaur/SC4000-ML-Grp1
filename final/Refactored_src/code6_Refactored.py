"""
Binary Classification for Outlier Detection using LightGBM
=========================================================
This script trains a LightGBM model to predict outliers using cross-validation.

Input Files:
1. train_c.csv (source: code1.py)
2. test_c.csv (source: code1.py)
3. feature_correlation_only_outlier.csv (source: code2_withoutOutlier.py)
4. sample_submission.csv (source: /data/raw)

Output Files:
1. bestline_submission_outliers_likelihood2.csv
"""

import datetime
import gc
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import precision_score

def suppress_warnings():
    """Suppress common warnings"""
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', RuntimeWarning)

def lgb_f1_score(y_hat, data):
    """
    Custom F1 score evaluation function for LightGBM
    """
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def load_data(train_path, test_path):
    """
    Load train and test data
    """
    train = pd.read_csv(train_path)
    train['week'] = train['week'].astype('int32')
    
    test = pd.read_csv(test_path)
    test['week'] = test['week'].astype('int32')
    
    return train, test

def prepare_features(train, test):
    """
    Prepare features for training
    """
    # Extract target variable
    target = train['outliers']
    
    # Define features to use
    exclude_cols = [
        'card_id', 'target', 'first_active_month', 'outliers',
        'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
        'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max'
    ]
    
    features = [c for c in train.columns if c not in exclude_cols]
    categorical_feats = [c for c in features if 'feature_' in c]
    
    print('Features: {}'.format(len(features)))
    print('Categorical features: {}'.format(len(categorical_feats)))
    
    # Keep only selected features
    train_features = train[features]
    test_features = test[features]
    
    return train_features, test_features, target, features, categorical_feats

def filter_features_by_correlation(train, test, features, categorical_feats, corr_scores_path, threshold=10):
    """
    Filter features based on correlation scores
    """
    corr_scores_df = pd.read_csv(corr_scores_path)
    
    # Filter features by correlation score
    selected_features = []
    selected_categorical = []
    
    for _f in corr_scores_df.itertuples():
        if _f[2] >= threshold:  # split_score >= threshold
            selected_features.append(_f[1])
            
            if _f[1] in categorical_feats:
                selected_categorical.append(_f[1])
    
    print('Selected features: {}'.format(len(selected_features)))
    
    # Filter dataframes to include only selected features
    train_filtered = train[selected_features]
    test_filtered = test[selected_features]
    
    return train_filtered, test_filtered, selected_features, selected_categorical


def get_lgb_params():
    """
    Define LightGBM parameters for CPU usage
    """
    params = {
        'num_leaves': 31,
        'min_data_in_leaf': 32, 
        'objective': 'binary',
        'max_depth': -1,
        'learning_rate': 0.005,
        'min_child_samples': 20,
        'boosting': 'gbdt',
        'feature_fraction': 0.9,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'bagging_seed': 11,
        'metric': 'f1',
        'lambda_l1': 0.1,
        'nthread': 7,
        'verbosity': -1,
        'is_unbalance': 'true'
    }
    
    return params
    

def train_and_predict(train, test, target, features, categorical_feats, n_folds=7):
    """
    Train LightGBM model with KFold cross-validation and make predictions
    """
    # Initialize KFold
    folds = KFold(n_splits=n_folds, shuffle=False)
    
    # Initialize arrays for out-of-fold predictions and test predictions
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    
    # Initialize dataframe for feature importance
    feature_importance_df = pd.DataFrame()
    
    # Train and predict per fold
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("Fold n°{}".format(fold_))
        
        # Create LightGBM datasets
        trn_data = lgb.Dataset(
            train.iloc[trn_idx][features], 
            label=target.iloc[trn_idx], 
            categorical_feature=categorical_feats
        )
        
        val_data = lgb.Dataset(
            train.iloc[val_idx][features], 
            label=target.iloc[val_idx], 
            categorical_feature=categorical_feats
        )
        
        # Get parameters
        params = get_lgb_params()
        
        # Train model
        num_round = 10000
        clf = lgb.train(
            params, 
            trn_data, 
            num_round,
            valid_sets=[trn_data, val_data],
            feval=lgb_f1_score,
            callbacks=[lgb.early_stopping(stopping_rounds=400)]
        )
        
        # Predict on validation set
        oof[val_idx] = clf.predict(
            train.iloc[val_idx][features], 
            num_iteration=clf.best_iteration
        )
        
        # Record feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        # Predict on test set and accumulate
        predictions += clf.predict(
            test[features], 
            num_iteration=clf.best_iteration
        ) / folds.n_splits
    
    return oof, predictions, feature_importance_df

def save_submission(predictions, submission_path, output_path):
    """
    Create and save submission file
    """
    sample_submission = pd.read_csv(submission_path)
    sample_submission['target'] = predictions
    sample_submission.to_csv(output_path, index=False)
    print('Submission saved to: {}'.format(output_path))


def main():
    """Main function to orchestrate the model training and prediction"""
    # Suppress warnings
    suppress_warnings()
    
    # Define file paths based on your folder structure
    # These paths are relative to your current working directory
    train_path = 'train_c.csv'
    test_path = 'test_c.csv'
    corr_scores_path = 'feature_correlation_only_outlier.csv'
    submission_path = 'sample_submission.csv'  # You'll need to adjust this if it's elsewhere
    output_path = 'bestline_submission_outliers_likelihood2.csv'
    
    # Load data
    print('[INFO] Loading train and test data')
    train, test = load_data(train_path, test_path)
    
    # Prepare features
    print('[INFO] Preparing features')
    train_features, test_features, target, features, categorical_feats = prepare_features(train, test)
    
    # Print target distribution
    print('Target values: {}'.format(np.unique(target)))
    
    # Filter features by correlation score
    print('[INFO] Filtering features by correlation score')
    train_filtered, test_filtered, selected_features, selected_categorical = filter_features_by_correlation(
        train_features, test_features, features, categorical_feats, corr_scores_path, threshold=10
    )
    
    # Train and predict
    print('[INFO] Starting training with 7-fold cross-validation')
    oof, predictions, feature_importance_df = train_and_predict(
        train_filtered, test_filtered, target, selected_features, selected_categorical, n_folds=7
    )
    
    # Save submission
    print('[INFO] Creating submission file')
    save_submission(predictions, submission_path, output_path)
    
    print('[SUCCESS] Model training and prediction completed')

if __name__ == "__main__":
    main()