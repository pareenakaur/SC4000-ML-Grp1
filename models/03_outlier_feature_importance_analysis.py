#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Importance Analysis using Permutation Method
---------------------------------------------------
This script performs feature importance analysis using a permutation approach to identify
the most relevant features for predicting outliers in financial transaction data.
"""

import datetime
import gc
import sys
import time
import warnings
import argparse
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import precision_score

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage of a dataframe by downcasting numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Whether to print memory reduction information
        
    Returns:
        pd.DataFrame: Memory-optimized dataframe
    """
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
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def load_data(train_path, test_path):
    """
    Load and preprocess training and test data.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        
    Returns:
        tuple: (train_features, test_features, target, feature_names, categorical_features)
    """
    # Load data with memory optimization
    train = reduce_mem_usage(pd.read_csv(train_path))
    test = reduce_mem_usage(pd.read_csv(test_path))
    
    # Remove unnamed index column if present
    if "Unnamed: 0" in train.columns:
        train = train.drop("Unnamed: 0", axis=1)
    if "Unnamed: 0" in test.columns:
        test = test.drop("Unnamed: 0", axis=1)
    
    # Extract target and define features
    target = train['outliers']
    excluded_cols = [
        'card_id', 'target', 'first_active_month', 'outliers',
        'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
        'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max'
    ]
    features = [c for c in train.columns if c not in excluded_cols]
    categorical_feats = [c for c in features if 'feature_' in c]
    
    # Extract feature datasets
    train_features = train[features]
    test_features = test[features]
    
    return train_features, test_features, target, features, categorical_feats


def get_feature_importances(train, target, features, categorical_feats, shuffle=False, seed=None):
    """
    Calculate feature importances using LightGBM.
    
    Args:
        train (pd.DataFrame): Training features
        target (pd.Series): Target variable
        features (list): List of feature names
        categorical_feats (list): List of categorical feature names
        shuffle (bool): Whether to shuffle the target (for null importance)
        seed (int, optional): Random seed for shuffling
        
    Returns:
        pd.DataFrame: DataFrame with feature importances
    """
    # Shuffle target if required
    y = target.copy()
    if shuffle:
        # Shuffle the target
        y = pd.DataFrame(y).sample(frac=1.0).iloc[:, 0]
    
    # Convert 'week' column to int32 if it exists
    if 'week' in train.columns:
        train['week'] = train['week'].astype('int32')
    
    # Prepare LightGBM dataset
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False)
    
    # Set LightGBM parameters
    lgb_params = {
        'device': 'gpu', 
        'gpu_platform_id': 0, 
        'gpu_device_id': 0,
        'num_leaves': 31,
        'min_data_in_leaf': 32, 
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.005,
        "min_child_samples": 20,
        "boosting": "gbdt",
        "feature_fraction": 0.9,
        "bagging_freq": 1,
        "bagging_fraction": 0.9,
        "bagging_seed": 11,
        "metric": 'rmse',
        "lambda_l1": 0.1,
        "nthread": 8,
    }
    
    # Train LightGBM model
    clf = lgb.train(
        params=lgb_params, 
        train_set=dtrain, 
        num_boost_round=3500
    )
    
    # Create feature importance DataFrame
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))
    
    return imp_df


def generate_null_importances(train_features, target, features, categorical_feats, 
                              nb_runs=80, output_path=None):
    """
    Generate null importances by shuffling the target multiple times.
    
    Args:
        train_features (pd.DataFrame): Training features
        target (pd.Series): Target variable
        features (list): List of feature names
        categorical_feats (list): List of categorical feature names
        nb_runs (int): Number of permutation runs
        output_path (str, optional): Path to save null importances
        
    Returns:
        tuple: (actual_importances, null_importances)
    """
    # Get actual feature importances
    print("Calculating actual feature importances...")
    actual_imp_df = get_feature_importances(
        train=train_features,
        target=target,
        features=features,
        categorical_feats=categorical_feats,
        shuffle=False
    )
    
    # Generate null importances
    null_imp_df = pd.DataFrame()
    start = time.time()
    dsp = ''
    
    print(f"Generating null importances with {nb_runs} runs...")
    for i in range(nb_runs):
        # Get current run importances with shuffled target
        imp_df = get_feature_importances(
            train=train_features,
            target=target,
            features=features,
            categorical_feats=categorical_feats,
            shuffle=True
        )
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
    
    print("\nCompleted null importance generation")
    
    # Save null importances if output path is provided
    if output_path:
        null_imp_df.to_feather(output_path)
        print(f"Saved null importances to {output_path}")
    
    return actual_imp_df, null_imp_df


def display_importance_distributions(actual_imp_df, null_imp_df, feature, output_dir=None):
    """
    Display distributions of feature importances.
    
    Args:
        actual_imp_df (pd.DataFrame): Actual feature importances
        null_imp_df (pd.DataFrame): Null feature importances
        feature (str): Feature name to display
        output_dir (str, optional): Directory to save plots
    """
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(
        null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_split'].values, 
        label='Null importances'
    )
    ax.vlines(
        x=actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_split'].mean(), 
        ymin=0, 
        ymax=np.max(a[0]), 
        color='r',
        linewidth=10, 
        label='Real Target'
    )
    ax.legend()
    ax.set_title(f'Split Importance of {feature.upper()}', fontweight='bold')
    plt.xlabel(f'Null Importance (split) Distribution for {feature.upper()}')
    
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(
        null_imp_df.loc[null_imp_df['feature'] == feature, 'importance_gain'].values, 
        label='Null importances'
    )
    ax.vlines(
        x=actual_imp_df.loc[actual_imp_df['feature'] == feature, 'importance_gain'].mean(), 
        ymin=0, 
        ymax=np.max(a[0]), 
        color='r',
        linewidth=10, 
        label='Real Target'
    )
    ax.legend()
    ax.set_title(f'Gain Importance of {feature.upper()}', fontweight='bold')
    plt.xlabel(f'Null Importance (gain) Distribution for {feature.upper()}')
    
    # Save plot if output directory is provided
    if output_dir:
        output_path = Path(output_dir) / f"importance_dist_{feature}.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved importance distribution plot to {output_path}")
    
    plt.tight_layout()
    plt.show()


def calculate_correlation_scores(actual_imp_df, null_imp_df):
    """
    Calculate correlation scores for features.
    
    Args:
        actual_imp_df (pd.DataFrame): Actual feature importances
        null_imp_df (pd.DataFrame): Null feature importances
        
    Returns:
        pd.DataFrame: DataFrame with correlation scores
    """
    print("Calculating correlation scores...")
    correlation_scores = []
    
    for _f in actual_imp_df['feature'].unique():
        # Get gain importances
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        
        # Get split importances
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        
        correlation_scores.append((_f, split_score, gain_score))
    
    # Create DataFrame with correlation scores
    scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    
    return scores_df


def plot_top_features(scores_df, n_top=30, output_dir=None):
    """
    Plot top features based on correlation scores.
    
    Args:
        scores_df (pd.DataFrame): DataFrame with correlation scores
        n_top (int): Number of top features to display
        output_dir (str, optional): Directory to save plots
    """
    plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    sns.barplot(
        x='split_score', 
        y='feature', 
        data=scores_df.sort_values('split_score', ascending=False).head(n_top), 
        ax=ax
    )
    ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
    
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    sns.barplot(
        x='gain_score', 
        y='feature', 
        data=scores_df.sort_values('gain_score', ascending=False).head(n_top), 
        ax=ax
    )
    ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
    
    # Save plot if output directory is provided
    if output_dir:
        output_path = Path(output_dir) / f"top_{n_top}_features.png"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved top features plot to {output_path}")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the feature importance analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    # parser.add_argument('--train', type=str, default='train_c_improved.csv', 
                        # help='Path to training data')
    # parser.add_argument('--test', type=str, default='test_c_improved.csv',
                        # help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='/home/UG/aarushi003/SC4000-ML-Grp1/data',
                        help='Directory to save outputs')
    parser.add_argument('--null_runs', type=int, default=80,
                        help='Number of permutation runs for null importance')
    parser.add_argument('--top_n', type=int, default=30,
                        help='Number of top features to display')
    parser.add_argument('--example_features', type=str, nargs='+', 
                        default=['hist_category_1_mean', 'category1_hist_weekend_ratio', 'new_month_nunique'],
                        help='Example features to display distributions')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # print(f"Loading data from {args.train} and {args.test}...")
    train_features, test_features, target, features, categorical_feats = load_data(
        "/home/UG/aarushi003/SC4000-ML-Grp1/data/processed/train_c_improved.csv", "/home/UG/aarushi003/SC4000-ML-Grp1/data/processed/test_c_improved.csv"
    )
    
    # Generate importances
    null_imp_path = r"/home/UG/aarushi003/SC4000-ML-Grp1/data/check_point/correlation_without_outlier_improved.feather"
    actual_imp_df, null_imp_df = generate_null_importances(
        train_features, target, features, categorical_feats, 
        nb_runs=args.null_runs, output_path=null_imp_path
    )
    
    # Display distributions for example features
    for feature in args.example_features:
        print(f"\nDisplaying importance distributions for feature: {feature}")
        display_importance_distributions(
            actual_imp_df, null_imp_df, feature, output_dir=args.output_dir
        )
    
    # Calculate correlation scores
    scores_df = calculate_correlation_scores(actual_imp_df, null_imp_df)
    print(f"Found {len(scores_df)} features with correlation scores")
    
    # Plot top features
    plot_top_features(scores_df, n_top=args.top_n, output_dir=args.output_dir)
    
    # Save correlation scores
    output_csv = "/home/UG/aarushi003/SC4000-ML-Grp1/data/processed/feature_correlation_only_outlier_improved.csv"
    scores_df.to_csv(output_csv, index=False)
    print(f"Saved correlation scores to {output_csv}")
    
    # Clean up memory
    gc.collect()


if __name__ == "__main__":
    main()