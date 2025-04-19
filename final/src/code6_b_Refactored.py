"""
Regression Model Training without Outliers - Alternative Implementation
======================================================================
This script trains a LightGBM regression model on non-outlier data with
alternative implementation approaches.

Input Files:
1. train_c.csv (source: code1.py)
2. test_c.csv (source: code1.py)
3. feature_correlation_without_outlier.csv (source: code4_withoutOutlier.py)
4. sample_submission.csv (source: data/raw)

Output Files:
1. bestline_submission_without_outliers.csv
"""

import datetime
import functools
import gc
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import precision_score


# Using decorator pattern for warning suppression
def with_warnings_suppressed(func):
    """Decorator to suppress warnings during function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Save original warning settings
        warning_settings = {
            'future': warnings.filters[0] if warnings.filters else None,
            'user': warnings.filters[1] if len(warnings.filters) > 1 else None,
            'deprecation': warnings.filters[2] if len(warnings.filters) > 2 else None,
            'runtime': warnings.filters[3] if len(warnings.filters) > 3 else None
        }
        
        # Suppress warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        
        try:
            # Execute the wrapped function
            return func(*args, **kwargs)
        finally:
            # Restore original warning settings
            if warning_settings['future']:
                warnings.filters[0] = warning_settings['future']
            if warning_settings['user']:
                warnings.filters[1] = warning_settings['user']
            if warning_settings['deprecation']:
                warnings.filters[2] = warning_settings['deprecation']
            if warning_settings['runtime']:
                warnings.filters[3] = warning_settings['runtime']
    
    return wrapper


# Using dataclass for configuration
@dataclass
class ModelConfig:
    """Configuration for model training"""
    # File paths
    train_path: str
    test_path: str
    corr_scores_path: str
    submission_path: str
    output_path: str
    
    # Model parameters
    n_folds: int = 7
    feature_threshold: int = 70
    num_boost_round: int = 10000
    early_stopping_rounds: int = 400
    
    # LightGBM parameters
    lgb_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default LightGBM parameters if none provided"""
        if not self.lgb_params:
            self.lgb_params = {
                'device': 'cpu',
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
                'nthread': 7,
                'verbosity': -1
            }


# Using a memory optimizer class with context manager
class MemoryOptimizer:
    """Context manager for memory optimization of pandas dataframes"""
    def __init__(self, df, verbose=True):
        self.df = df
        self.verbose = verbose
        self.start_mem = None
        
    def __enter__(self):
        """Start the context manager and record initial memory usage"""
        self.start_mem = self.df.memory_usage().sum() / 1024**2
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the context manager and print memory usage reduction"""
        end_mem = self.df.memory_usage().sum() / 1024**2
        if self.verbose and self.start_mem > end_mem:
            reduction = 100 * (self.start_mem - end_mem) / self.start_mem
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, reduction))
    
    def optimize(self):
        """Optimize memory usage by downcasting numeric columns"""
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        
        for col in self.df.columns:
            col_type = self.df[col].dtypes
            
            if col_type in numerics:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                
                # Downcast integers
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)  
                # Downcast floats
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
        
        return self.df


# Using a class-based approach for data management
class DataManager:
    """Class to manage data loading, preprocessing, and feature selection"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.train = None
        self.test = None
        self.target = None
        self.features = None
        self.categorical_features = None
        self.selected_features = None
        self.selected_categorical = None
    
    def load_data(self):
        """Load train and test data with memory optimization"""
        print('[INFO] Loading and optimizing train data')
        train_df = pd.read_csv(self.config.train_path)
        with MemoryOptimizer(train_df) as optimizer:
            self.train = optimizer.optimize()
        
        print('[INFO] Loading and optimizing test data')
        test_df = pd.read_csv(self.config.test_path)
        with MemoryOptimizer(test_df) as optimizer:
            self.test = optimizer.optimize()
        
        # Filter out outliers
        self.train = self.train[self.train['outliers'] == 0]
        
        return self
    
    def prepare_features(self):
        """Extract target and prepare features"""
        # Extract target variable
        self.target = self.train['target']
        
        # Define excluded columns
        exclude_cols = [
            'card_id', 'target', 'first_active_month', 'outliers',
            'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
            'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max'
        ]
        
        # Define features
        self.features = [c for c in self.train.columns if c not in exclude_cols]
        self.categorical_features = [c for c in self.features if 'feature_' in c]
        
        # Filter columns
        self.train = self.train[self.features]
        self.test = self.test[self.features]
        
        print(f'[INFO] Prepared {len(self.features)} features ({len(self.categorical_features)} categorical)')
        
        return self
    
    def select_features_by_correlation(self):
        """Select features based on correlation scores"""
        # Load correlation scores
        corr_scores_df = pd.read_csv(self.config.corr_scores_path)
        
        # Initialize feature lists
        self.selected_features = []
        self.selected_categorical = []
        
        # Select features based on threshold
        for row in corr_scores_df.itertuples():
            feature_name = row[1]
            split_score = row[2]
            
            if split_score >= self.config.feature_threshold:
                self.selected_features.append(feature_name)
                
                if feature_name in self.categorical_features:
                    self.selected_categorical.append(feature_name)
        
        # Filter dataframes
        self.train = self.train[self.selected_features]
        self.test = self.test[self.selected_features]
        
        print(f'[INFO] Selected {len(self.selected_features)} features with correlation threshold {self.config.feature_threshold}')
        
        return self


# Using a class-based approach for model training
class LightGBMTrainer:
    """Class to handle LightGBM model training and prediction"""
    def __init__(self, config: ModelConfig, data_manager: DataManager):
        self.config = config
        self.data = data_manager
        self.oof_predictions = None
        self.test_predictions = None
        self.feature_importance = None
        self.cv_score = None
    
    def train_and_predict(self):
        """Train model with K-fold cross-validation and make predictions"""
        # Initialize cross-validation
        folds = KFold(n_splits=self.config.n_folds, shuffle=False)
        
        # Initialize arrays
        self.oof_predictions = np.zeros(len(self.data.train))
        self.test_predictions = np.zeros(len(self.data.test))
        self.feature_importance = []
        
        # Train model for each fold
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.data.train.values, self.data.target.values)):
            print("Fold n°{}".format(fold_))
            
            # Create training and validation datasets
            train_set = lgb.Dataset(
                self.data.train.iloc[trn_idx][self.data.selected_features],
                label=self.data.target.iloc[trn_idx],
                categorical_feature=self.data.selected_categorical
            )
            
            valid_set = lgb.Dataset(
                self.data.train.iloc[val_idx][self.data.selected_features],
                label=self.data.target.iloc[val_idx],
                categorical_feature=self.data.selected_categorical
            )
            
            # Train model
            callbacks = [lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds)]
            model = lgb.train(
                self.config.lgb_params,
                train_set,
                num_boost_round=self.config.num_boost_round,
                valid_sets=[train_set, valid_set],
                callbacks=callbacks
            )
            
            # Make predictions
            self.oof_predictions[val_idx] = model.predict(
                self.data.train.iloc[val_idx][self.data.selected_features],
                num_iteration=model.best_iteration
            )
            
            fold_preds = model.predict(
                self.data.test[self.data.selected_features],
                num_iteration=model.best_iteration
            )
            self.test_predictions += fold_preds / self.config.n_folds
            
            # Record feature importance
            fold_importance = pd.DataFrame({
                'feature': self.data.selected_features,
                'importance': model.feature_importance(),
                'fold': fold_ + 1
            })
            self.feature_importance.append(fold_importance)
        
        # Combine feature importance from all folds
        self.feature_importance = pd.concat(self.feature_importance, axis=0)
        
        # Calculate CV score
        self.cv_score = mean_squared_error(self.oof_predictions, self.data.target) ** 0.5
        print("CV score: {:<8.5f}".format(self.cv_score))
        
        return self
    
    def save_submission(self):
        """Save predictions to submission file"""
        submission = pd.read_csv(self.config.submission_path)
        submission['target'] = self.test_predictions
        submission.to_csv(self.config.output_path, index=False)
        print(f'[INFO] Submission saved to: {self.config.output_path}')
        return self


# Using a command pattern for pipeline execution
class ModelPipeline:
    """Pipeline to orchestrate the entire model training process"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_manager = None
        self.trainer = None
    
    def execute(self):
        """Execute the entire pipeline"""
        # Initialize data manager and load data
        self.data_manager = DataManager(self.config)
        self.data_manager.load_data().prepare_features().select_features_by_correlation()
        
        # Initialize trainer and train model
        self.trainer = LightGBMTrainer(self.config, self.data_manager)
        self.trainer.train_and_predict().save_submission()
        
        print('[SUCCESS] Model pipeline completed successfully')
        return self


@with_warnings_suppressed
def main():
    """Main function"""
    # Create configuration
    config = ModelConfig(
        train_path='train_c.csv',
        test_path='test_c.csv',
        corr_scores_path='feature_correlation_without_outlier.csv',
        submission_path='sample_submission.csv',
        output_path='bestline_submission_without_outliers.csv',
        n_folds=7,
        feature_threshold=70
    )
    
    # Execute pipeline
    pipeline = ModelPipeline(config)
    pipeline.execute()


if __name__ == "__main__":
    main()