"""
Memory-Optimized Enhanced Regression Model Training without Outliers
====================================================================
This script trains an ensemble of regression models on non-outlier data
with memory-efficient feature engineering and hyperparameter optimization.

Input Files:
1. train_c.csv (source: code1.py)
2. test_c.csv (source: code1.py)
3. feature_correlation_without_outlier.csv (source: code4_withoutOutlier.py)
4. sample_submission.csv (source: data/raw)

Output Files:
1. enhanced_submission_without_outliers.csv
"""

import datetime
import functools
import gc
import os
import sys
import time
import warnings
import pickle
import random
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
import xgboost as xgb

try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("BayesianOptimization not available. Will use default hyperparameters.")


# Memory monitoring function
def get_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Using decorator pattern for warning suppression
def with_warnings_suppressed(func):
    """Decorator to suppress warnings during function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Suppress warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter('ignore', UserWarning)
        warnings.simplefilter('ignore', DeprecationWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        
        try:
            # Execute the wrapped function
            return func(*args, **kwargs)
        finally:
            # No need to restore warning settings as they're not that important
            pass
    
    return wrapper


# Using dataclass for configuration with memory-optimized parameters
@dataclass
class ModelConfig:
    """Memory-efficient configuration for model training"""
    # File paths
    train_path: str
    test_path: str
    corr_scores_path: str
    submission_path: str
    output_path: str
    
    # Model parameters
    n_folds: int = 7
    feature_threshold: int = 50
    num_boost_round: int = 10000
    early_stopping_rounds: int = 200
    random_seed: int = 42
    
    # Feature engineering controls
    use_feature_engineering: bool = True
    max_engineered_features: int = 50  # Limit number of engineered features
    max_categorical_stats: int = 5     # Limit number of categorical features to generate stats for
    max_interaction_features: int = 3  # Limit number of features for interactions
    
    # Model settings
    use_ensemble: bool = True
    use_hyperopt: bool = False  # Disabled by default to save memory
    
    # Ensemble weights
    lgb_weight: float = 0.65
    xgb_weight: float = 0.35
    
    # LightGBM parameters (optimized for memory efficiency)
    lgb_params: Dict = field(default_factory=dict)
    
    # XGBoost parameters
    xgb_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default parameters if none provided"""
        # Default LightGBM parameters
        if not self.lgb_params:
            self.lgb_params = {
                'objective': 'regression',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.01,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'metric': 'rmse',
                'nthread': -1,
                'verbose': -1
            }
        
        # Default XGBoost parameters
        if not self.xgb_params:
            self.xgb_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': 0.01,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'alpha': 0.1,
                'lambda': 1.0,
                'min_child_weight': 1,
                'nthread': -1,
                'verbosity': 0
            }

        # Set random seeds
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)


# Memory-efficient memory optimizer
class MemoryOptimizer:
    """Memory-efficient optimizer for pandas dataframes"""
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
        """Optimize memory usage with enhanced type conversion"""
        # Second pass: optimize numeric columns
        for col in self.df.columns:
            col_type = self.df[col].dtypes
            
            # Handle numeric types
            if pd.api.types.is_numeric_dtype(col_type):
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                
                # Handle integers
                if pd.api.types.is_integer_dtype(col_type):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                
                # Handle floats
                elif pd.api.types.is_float_dtype(col_type):
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
            
            # Handle categorical and object types
            elif pd.api.types.is_object_dtype(col_type):
                if self.df[col].nunique() < 0.5 * len(self.df):  # If fewer unique values than 50% of rows
                    self.df[col] = self.df[col].astype('category')
        
        # Force garbage collection
        gc.collect()
        
        return self.df


# Memory-efficient feature engineer
class MemoryEfficientFeatureEngineer:
    """Memory-efficient class for feature engineering"""
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def engineer_features(self, train_df, test_df, categorical_features, target=None):
        """Apply memory-efficient feature engineering to train and test dataframes"""
        print("[INFO] Applying memory-efficient feature engineering")
        
        # Track memory usage
        start_mem = get_memory_usage()
        print(f"[INFO] Memory usage at start: {start_mem:.2f} MB")
        
        # Make copies to avoid modifying originals
        train = train_df.copy()
        test = test_df.copy()
        
        # 1. Select top categorical features for stats (to limit memory usage)
        if categorical_features:
            if len(categorical_features) > self.config.max_categorical_stats:
                if target is not None:
                    # Select categorical features with highest mutual information
                    cat_mi_scores = {}
                    for cat in categorical_features:
                        if cat in train.columns:
                            # Convert to numeric for MI calculation
                            train_cat_numeric = train[cat].astype('category').cat.codes
                            cat_mi_scores[cat] = mutual_info_regression(
                                train_cat_numeric.values.reshape(-1, 1), 
                                target.values
                            )[0]
                    
                    # Sort by MI score
                    sorted_cats = sorted(cat_mi_scores.items(), key=lambda x: x[1], reverse=True)
                    selected_cats = [cat for cat, _ in sorted_cats[:self.config.max_categorical_stats]]
                else:
                    # If no target, select features with moderate cardinality
                    cat_nunique = {}
                    for cat in categorical_features:
                        if cat in train.columns:
                            cat_nunique[cat] = train[cat].nunique()
                    
                    # Sort by number of unique values (prefer moderate cardinality)
                    sorted_cats = sorted(cat_nunique.items(), key=lambda x: x[1])
                    # Select from middle - not too many or too few unique values
                    middle_idx = len(sorted_cats) // 2
                    start_idx = max(0, middle_idx - self.config.max_categorical_stats // 2)
                    selected_cats = [cat for cat, _ in sorted_cats[start_idx:start_idx + self.config.max_categorical_stats]]
            else:
                selected_cats = categorical_features
                
            print(f"[INFO] Selected {len(selected_cats)} categorical features for generating statistics")
            
            # 2. Generate statistical features for selected categorical features (one at a time to save memory)
            for cat_feat in selected_cats:
                if cat_feat in train.columns:
                    print(f"[INFO] Generating statistics for {cat_feat}")
                    
                    # Select numeric columns for aggregation
                    numeric_cols = [col for col in train.columns 
                                   if col != cat_feat 
                                   and col not in categorical_features 
                                   and pd.api.types.is_numeric_dtype(train[col].dtype)]
                    
                    # Only use a subset of numeric columns if there are too many
                    if len(numeric_cols) > 5:
                        if target is not None:
                            # Select columns most correlated with target
                            corrs = {}
                            for col in numeric_cols:
                                corrs[col] = abs(np.corrcoef(train[col].values, target.values)[0, 1])
                            numeric_cols = [col for col, _ in sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]]
                        else:
                            # Just take first 5
                            numeric_cols = numeric_cols[:5]
                    
                    # Create group statistics only for mean to reduce memory
                    train_groups = train.groupby(cat_feat)[numeric_cols].mean().reset_index()
                    train_groups.columns = [cat_feat] + [f'{cat_feat}_{col}_mean' for col in numeric_cols]
                    
                    # Merge statistics back to dataframes
                    train = pd.merge(train, train_groups, on=cat_feat, how='left')
                    test = pd.merge(test, train_groups, on=cat_feat, how='left')
                    
                    # Fill NaN values in test set for any categories not seen in training
                    for col in train_groups.columns:
                        if col != cat_feat and col in test.columns:
                            test[col].fillna(train[col].mean(), inplace=True)
                    
                    # Force garbage collection
                    gc.collect()
                    
                    print(f"[INFO] Memory usage after {cat_feat} features: {get_memory_usage():.2f} MB")
        
        # 3. Feature scaling - replace with simple normalization to save memory
        # Select numerical features
        numerical_features = [f for f in train.columns if f not in categorical_features 
                             and f in test.columns and pd.api.types.is_numeric_dtype(train[f].dtype)]
        
        if numerical_features:
            print("[INFO] Normalizing numerical features")
            # Simple normalization (min-max scaling)
            for col in numerical_features:
                col_min = train[col].min()
                col_max = train[col].max()
                if col_max > col_min:
                    train[col] = (train[col] - col_min) / (col_max - col_min)
                    test[col] = (test[col] - col_min) / (col_max - col_min)
            
            # Force garbage collection
            gc.collect()
        
        # 4. Feature interactions (limited to save memory)
        if numerical_features:
            # Find top correlated features with target if target is provided
            if target is not None and len(numerical_features) > self.config.max_interaction_features:
                correlations = {}
                for col in numerical_features:
                    if col in train.columns:
                        correlations[col] = abs(np.corrcoef(train[col].values, target.values)[0, 1])
                
                # Sort features by correlation
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                top_features = [f[0] for f in sorted_features[:self.config.max_interaction_features]]
            else:
                # If no target or few features, use first N numerical features
                top_features = numerical_features[:min(len(numerical_features), self.config.max_interaction_features)]
            
            # Generate a limited set of polynomial features
            print("[INFO] Generating limited feature interactions")
            for i, f1 in enumerate(top_features):
                for j, f2 in enumerate(top_features[i+1:], i+1):
                    # Multiply features
                    train[f'{f1}_mult_{f2}'] = train[f1] * train[f2]
                    test[f'{f1}_mult_{f2}'] = test[f1] * test[f2]
                    
                    # Division (prevent division by zero)
                    train[f'{f1}_div_{f2}'] = train[f1] / (train[f2] + 1e-8)
                    test[f'{f1}_div_{f2}'] = test[f1] / (test[f2] + 1e-8)
                    
                    # Force garbage collection after each pair
                    gc.collect()
        
        # 5. Add only essential aggregate features
        if len(numerical_features) > 1:
            print("[INFO] Creating minimal aggregate features")
            
            # Mean of all numerical features
            train['mean_all_num'] = train[numerical_features].mean(axis=1)
            test['mean_all_num'] = test[numerical_features].mean(axis=1)
            
            # Standard deviation of all numerical features
            train['std_all_num'] = train[numerical_features].std(axis=1)
            test['std_all_num'] = test[numerical_features].std(axis=1)
            
            # Force garbage collection
            gc.collect()
        
        # Ensure we don't have too many features (memory concerns)
        if train.shape[1] > self.config.max_engineered_features:
            print(f"[INFO] Too many features ({train.shape[1]}), reducing to {self.config.max_engineered_features}")
            
            # If target is available, use mutual information to select features
            if target is not None:
                print("[INFO] Selecting features using mutual information")
                numeric_cols = [col for col in train.columns if pd.api.types.is_numeric_dtype(train[col].dtype)]
                
                # Calculate mutual information for numeric features
                mi_scores = mutual_info_regression(train[numeric_cols], target)
                mi_features = pd.Series(mi_scores, index=numeric_cols)
                sorted_features = mi_features.sort_values(ascending=False)
                
                # Get top N features
                top_features = sorted_features.index[:self.config.max_engineered_features].tolist()
                
                # Make sure to include categorical features if possible
                remaining_slots = self.config.max_engineered_features - len(top_features)
                if remaining_slots > 0:
                    cat_features_to_add = [f for f in categorical_features if f not in top_features][:remaining_slots]
                    top_features.extend(cat_features_to_add)
                
                # Filter datasets
                train = train[top_features]
                test = test[top_features]
            else:
                # Without target, just keep the first N features
                keep_features = train.columns[:self.config.max_engineered_features].tolist()
                train = train[keep_features]
                test = test[keep_features]
        
        # Final memory usage
        end_mem = get_memory_usage()
        print(f"[INFO] Memory usage after feature engineering: {end_mem:.2f} MB (change: {end_mem - start_mem:.2f} MB)")
        
        # Force final garbage collection
        gc.collect()
        
        return train, test


# Data Manager
class DataManager:
    """Base class to manage data loading, preprocessing, and feature selection"""
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


# Memory-Efficient Data Manager
class MemoryEfficientDataManager(DataManager):
    """Memory-efficient class to manage data processing and feature selection"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_engineer = MemoryEfficientFeatureEngineer(config)
        self.train_engineered = None
        self.test_engineered = None
    
    def prepare_features(self):
        """Extract target and prepare features with enhanced memory efficiency"""
        # Extract target variable
        self.target = self.train['target']
        
        # Define excluded columns (anything that looks like a date)
        exclude_cols = [
            'card_id', 'target', 'first_active_month', 'outliers',
            'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
            'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max'
        ]
        
        # Add any columns with 'date' in name to exclude
        date_cols = [col for col in self.train.columns if 'date' in col.lower()]
        exclude_cols.extend(date_cols)
        
        # Define features
        self.features = [c for c in self.train.columns if c not in exclude_cols]
        
        # Basic categorical feature detection
        self.categorical_features = [c for c in self.features if 'feature_' in c]
        
        # Check for additional categorical features (limited check to save memory)
        for col in self.features:
            if col not in self.categorical_features:
                # Check if the column is object type or has few unique values
                if self.train[col].dtype == 'object' or self.train[col].nunique() < 20:
                    self.categorical_features.append(col)
                    print(f'[INFO] Detected additional categorical feature: {col}')
                    
                    # Force garbage collection
                    gc.collect()
        
        # Filter columns
        self.train = self.train[self.features]
        self.test = self.test[self.features]
        
        print(f'[INFO] Prepared {len(self.features)} features ({len(self.categorical_features)} categorical)')
        
        # Apply feature engineering if enabled
        if self.config.use_feature_engineering:
            # Use smaller chunks to process if we have many features
            if len(self.features) > 100:
                print("[INFO] Processing features in chunks to save memory")
                
                # Process in chunks of 100 features
                chunk_size = 100
                feature_chunks = [self.features[i:i + chunk_size] for i in range(0, len(self.features), chunk_size)]
                
                # Process each chunk
                train_parts = []
                test_parts = []
                
                for i, chunk in enumerate(feature_chunks):
                    print(f"[INFO] Processing chunk {i+1}/{len(feature_chunks)}")
                    
                    # Filter categorical features for this chunk
                    chunk_categorical = [f for f in self.categorical_features if f in chunk]
                    
                    # Engineer features for this chunk
                    train_chunk = self.train[chunk]
                    test_chunk = self.test[chunk]
                    
                    train_eng, test_eng = self.feature_engineer.engineer_features(
                        train_chunk, test_chunk, chunk_categorical, self.target
                    )
                    
                    train_parts.append(train_eng)
                    test_parts.append(test_eng)
                    
                    # Force garbage collection
                    gc.collect()
                
                # Combine processed chunks
                self.train_engineered = pd.concat(train_parts, axis=1)
                self.test_engineered = pd.concat(test_parts, axis=1)
                
                # Remove duplicates if any
                self.train_engineered = self.train_engineered.loc[:, ~self.train_engineered.columns.duplicated()]
                self.test_engineered = self.test_engineered.loc[:, ~self.test_engineered.columns.duplicated()]
            else:
                # Regular processing for smaller feature sets
                self.train_engineered, self.test_engineered = self.feature_engineer.engineer_features(
                    self.train, self.test, self.categorical_features, self.target
                )
            
            # Update features list with new engineered features
            self.features = list(self.train_engineered.columns)
            print(f'[INFO] After engineering: {len(self.features)} features')
        else:
            self.train_engineered, self.test_engineered = self.train.copy(), self.test.copy()
        
        return self
    
    def select_features_by_correlation(self):
        """Memory-efficient feature selection"""
        # Load correlation scores
        corr_scores_df = pd.read_csv(self.config.corr_scores_path)
        
        # Initialize feature lists
        self.selected_features = []
        self.selected_categorical = []
        
        # Select features based on threshold and ensure they exist in the dataframe
        for row in corr_scores_df.itertuples():
            feature_name = row[1]
            split_score = row[2]
            
            if split_score >= self.config.feature_threshold:
                # Only add if feature exists in our dataframe
                if feature_name in self.train_engineered.columns:
                    self.selected_features.append(feature_name)
                    
                    if feature_name in self.categorical_features:
                        self.selected_categorical.append(feature_name)
        
        # Add engineered features - only add a few important ones
        engineered_markers = ['_mult_', '_div_', 'mean_all_num', 'std_all_num']
        for feature in self.train_engineered.columns:
            if feature not in self.selected_features:
                if any(marker in feature for marker in engineered_markers):
                    self.selected_features.append(feature)
                    print(f'[INFO] Adding engineered feature: {feature}')
        
        # Limit total features if we have too many
        if len(self.selected_features) > 100:
            print(f"[INFO] Too many selected features ({len(self.selected_features)}), reducing to 100")
            
            # Keep correlation-selected features first, then engineered
            corr_features = [f for f in self.selected_features if not any(marker in f for marker in engineered_markers)]
            eng_features = [f for f in self.selected_features if any(marker in f for marker in engineered_markers)]
            
            # Prioritize corr features, then add engineered until we hit limit
            keep_corr = min(len(corr_features), 70)  # Keep up to 70 correlation features
            keep_eng = min(len(eng_features), 30)    # Keep up to 30 engineered features
            
            self.selected_features = corr_features[:keep_corr] + eng_features[:keep_eng]
            self.selected_categorical = [f for f in self.selected_categorical if f in self.selected_features]
        
        # Filter dataframes
        self.train = self.train_engineered[self.selected_features]
        self.test = self.test_engineered[self.selected_features]
        
        print(f'[INFO] Selected {len(self.selected_features)} features for modeling')
        
        # Force garbage collection
        gc.collect()
        
        return self


# Memory-efficient Model Trainer
class MemoryEfficientModelTrainer:
    """Memory-efficient class to handle model training and prediction"""
    def __init__(self, config: ModelConfig, data_manager: MemoryEfficientDataManager):
        self.config = config
        self.data = data_manager
        self.oof_predictions = None
        self.test_predictions = None
    
    def _train_lgb(self, train_idx, val_idx, fold_):
        """Train LightGBM model for a fold"""
        # Get train and validation data
        X_train = self.data.train.iloc[train_idx]
        y_train = self.data.target.iloc[train_idx]
        X_val = self.data.train.iloc[val_idx]
        y_val = self.data.target.iloc[val_idx]
        
        # Create LightGBM datasets
        train_set = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=self.data.selected_categorical
        )
        
        valid_set = lgb.Dataset(
            X_val,
            label=y_val,
            categorical_feature=self.data.selected_categorical
        )
        
        # Train model
        callbacks = [lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False)]
        model = lgb.train(
            self.config.lgb_params,
            train_set,
            num_boost_round=self.config.num_boost_round,
            valid_sets=[train_set, valid_set],
            callbacks=callbacks,
            verbose_eval=100
        )
        
        return model
    
    def _train_xgb(self, train_idx, val_idx, fold_):
        """Train XGBoost model for a fold"""
        # Get train and validation data
        X_train = self.data.train.iloc[train_idx]
        y_train = self.data.target.iloc[train_idx]
        X_val = self.data.train.iloc[val_idx]
        y_val = self.data.target.iloc[val_idx]
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        
        # Set up watchlist
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
        # Train model
        model = xgb.train(
            params=self.config.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.config.num_boost_round,
            evals=watchlist,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose_eval=100
        )
        
        return model
    
    def train_and_predict(self):
        """Train models with K-fold cross-validation and make predictions"""
        # Initialize cross-validation
        folds = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
        
        # Initialize arrays
        n_samples = len(self.data.train)
        n_test_samples = len(self.data.test)
        
        self.oof_predictions = {
            'lgb': np.zeros(n_samples),
            'xgb': np.zeros(n_samples) if self.config.use_ensemble else None,
            'ensemble': np.zeros(n_samples)
        }
        
        self.test_predictions = {
            'lgb': np.zeros(n_test_samples),
            'xgb': np.zeros(n_test_samples) if self.config.use_ensemble else None,
            'ensemble': np.zeros(n_test_samples)
        }
        
        # Train and predict for each fold
        for fold_, (train_idx, val_idx) in enumerate(folds.split(self.data.train)):
            print(f"\n=========== Fold {fold_+1}/{self.config.n_folds} ===========")
            
            # Train LightGBM
            print(f"Training LightGBM model...")
            lgb_model = self._train_lgb(train_idx, val_idx, fold_)
            
            # Make LightGBM predictions
            self.oof_predictions['lgb'][val_idx] = lgb_model.predict(
                self.data.train.iloc[val_idx],
                num_iteration=lgb_model.best_iteration
            )
            
            # Free memory by making test predictions and deleting the model
            lgb_test_preds = lgb_model.predict(
                self.data.test,
                num_iteration=lgb_model.best_iteration
            )
            self.test_predictions['lgb'] += lgb_test_preds / self.config.n_folds
            
            # Delete model to free memory
            del lgb_model
            gc.collect()
            
            # Train XGBoost if ensemble is enabled
            if self.config.use_ensemble:
                print(f"Training XGBoost model...")
                xgb_model = self._train_xgb(train_idx, val_idx, fold_)
                
                # Make XGBoost predictions
                X_val = xgb.DMatrix(self.data.train.iloc[val_idx])
                X_test = xgb.DMatrix(self.data.test)
                
                self.oof_predictions['xgb'][val_idx] = xgb_model.predict(
                    X_val, 
                    ntree_limit=xgb_model.best_ntree_limit
                )
                
                xgb_test_preds = xgb_model.predict(
                    X_test,
                    ntree_limit=xgb_model.best_ntree_limit
                )
                self.test_predictions['xgb'] += xgb_test_preds / self.config.n_folds
                
                # Delete model to free memory
                del xgb_model, X_val, X_test
                gc.collect()
            
            # Calculate ensemble predictions
            if self.config.use_ensemble:
                # For validation set
                self.oof_predictions['ensemble'][val_idx] = (
                    self.config.lgb_weight * self.oof_predictions['lgb'][val_idx] +
                    self.config.xgb_weight * self.oof_predictions['xgb'][val_idx]
                )
                
                # For test set
                fold_ensemble_preds = (
                    self.config.lgb_weight * lgb_test_preds +
                    self.config.xgb_weight * xgb_test_preds
                )
                self.test_predictions['ensemble'] += fold_ensemble_preds / self.config.n_folds
            else:
                self.oof_predictions['ensemble'][val_idx] = self.oof_predictions['lgb'][val_idx]
                self.test_predictions['ensemble'] = self.test_predictions['lgb']
            
            # Calculate RMSE for this fold
            fold_rmse = np.sqrt(mean_squared_error(
                self.data.target.iloc[val_idx], 
                self.oof_predictions['ensemble'][val_idx]
            ))
            print(f"Fold {fold_+1} - RMSE: {fold_rmse:.6f}")
            
            # Free memory
            gc.collect()
        
        # Calculate overall metrics
        rmse = np.sqrt(mean_squared_error(self.data.target, self.oof_predictions['ensemble']))
        mae = mean_absolute_error(self.data.target, self.oof_predictions['ensemble'])
        r2 = r2_score(self.data.target, self.oof_predictions['ensemble'])
        
        print("\n=========== Final Metrics ===========")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        return self
    
    def save_submission(self):
        """Save predictions to submission file"""
        submission = pd.read_csv(self.config.submission_path)
        submission['target'] = self.test_predictions['ensemble']
        submission.to_csv(self.config.output_path, index=False)
        print(f'[INFO] Submission saved to: {self.config.output_path}')
        return self


# Memory-efficient pipeline
class MemoryEfficientPipeline:
    """Memory-efficient pipeline to orchestrate the entire model training process"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_manager = None
        self.trainer = None
    
    def execute(self):
        """Execute the enhanced pipeline"""
        # Initialize data manager and load data
        self.data_manager = MemoryEfficientDataManager(self.config)
        self.data_manager.load_data().prepare_features().select_features_by_correlation()
        
        # Initialize trainer and train model
        self.trainer = MemoryEfficientModelTrainer(self.config, self.data_manager)
        self.trainer.train_and_predict().save_submission()
        
        print('[SUCCESS] Memory-efficient model pipeline completed successfully')
        return self


@with_warnings_suppressed
def main():
    """Main function"""
    # Create configuration optimized for memory efficiency
    config = ModelConfig(
        train_path='train_c.csv',
        test_path='test_c.csv',
        corr_scores_path='feature_correlation_without_outlier.csv',
        submission_path='sample_submission.csv',
        output_path='memory_efficient_submission_without_outliers.csv',
        n_folds=7,
        feature_threshold=50,
        num_boost_round=10000,
        early_stopping_rounds=200,
        random_seed=42,
        use_feature_engineering=True,
        max_engineered_features=50,
        max_categorical_stats=5,
        max_interaction_features=3,
        use_ensemble=True,
        use_hyperopt=False  # Disabled to save memory
    )
    
    # Execute memory-efficient pipeline
    pipeline = MemoryEfficientPipeline(config)
    pipeline.execute()


if __name__ == "__main__":
    try:
        import psutil
    except ImportError:
        # Create simple function if psutil is not available
        def get_memory_usage():
            """Fallback function"""
            return 0
        
        # Attach to globals
        globals()['get_memory_usage'] = get_memory_usage
        print("Note: psutil not available, memory tracking will be disabled.")
    
    main()