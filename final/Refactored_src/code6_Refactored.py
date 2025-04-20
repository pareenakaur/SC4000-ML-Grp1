import datetime
import gc
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

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

def prepare_features(train, test, perform_engineering=True):
    """
    Prepare features for training with enhanced feature engineering
    """
    # Extract target variable
    target = train['outliers']
    
    # Define features to use
    exclude_cols = [
        'card_id', 'target', 'first_active_month', 'outliers',
        'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
        'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max'
    ]
    
    # Find date-like columns and add them to exclude_cols
    for col in train.columns:
        if col not in exclude_cols:
            # Check if column has date-like strings
            if train[col].dtype == 'object':
                try:
                    # Try parsing a sample value as date
                    sample_val = train[col].iloc[0]
                    if isinstance(sample_val, str) and (
                        '-' in sample_val or '/' in sample_val or 
                        sample_val.count(':') >= 2
                    ):
                        # Skip columns that look like dates or timestamps
                        exclude_cols.append(col)
                        print(f"Excluding possible date column: {col}")
                except:
                    pass
    
    features = [c for c in train.columns if c not in exclude_cols]
    categorical_feats = [c for c in features if 'feature_' in c]
    
    print('Features: {}'.format(len(features)))
    print('Categorical features: {}'.format(len(categorical_feats)))
    
    # Keep only selected features
    train_features = train[features].copy()
    test_features = test[features].copy()
    
    if perform_engineering:
        print("Performing advanced feature engineering...")
        
        # Handle non-numeric features
        for col in train_features.columns:
            if train_features[col].dtype == 'object':
                print(f"Converting non-numeric column to category: {col}")
                train_features[col] = train_features[col].astype('category').cat.codes
                test_features[col] = test_features[col].astype('category').cat.codes
                
                # Add to categorical features if not already there
                if col not in categorical_feats:
                    categorical_feats.append(col)
        
        # Feature scaling for numerical features
        numerical_feats = [f for f in features if f not in categorical_feats]
        scaler = StandardScaler()
        
        if numerical_feats:
            train_features[numerical_feats] = scaler.fit_transform(train_features[numerical_feats])
            test_features[numerical_feats] = scaler.transform(test_features[numerical_feats])
        
        # Create feature interactions for top correlated features
        # Use direct correlation calculation on numeric columns
        numeric_cols = train_features.select_dtypes(include=np.number).columns.tolist()
        correlations = pd.Series(index=numeric_cols)
        
        for col in numeric_cols:
            correlations[col] = np.abs(np.corrcoef(train_features[col], target)[0, 1])
        
        top_features = correlations.sort_values(ascending=False).head(5).index.tolist()
        
        # Create pairwise interactions between top features
        for i, f1 in enumerate(top_features):
            for f2 in top_features[i+1:]:
                feat_name = f"{f1}_mult_{f2}"
                train_features[feat_name] = train_features[f1] * train_features[f2]
                test_features[feat_name] = test_features[f1] * test_features[f2]
                
                feat_name = f"{f1}_div_{f2}"
                train_features[feat_name] = train_features[f1] / (train_features[f2] + 1e-5)
                test_features[feat_name] = test_features[f1] / (test_features[f2] + 1e-5)
                
                feat_name = f"{f1}_plus_{f2}"
                train_features[feat_name] = train_features[f1] + train_features[f2]
                test_features[feat_name] = test_features[f1] + test_features[f2]
                
                feat_name = f"{f1}_minus_{f2}"
                train_features[feat_name] = train_features[f1] - train_features[f2]
                test_features[feat_name] = test_features[f1] - test_features[f2]
        
        # Update feature list
        features = train_features.columns.tolist()
    
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
            feature_name = _f[1]
            if feature_name in train.columns:  # Check if feature exists in the dataframe
                selected_features.append(feature_name)
                
                if feature_name in categorical_feats:
                    selected_categorical.append(feature_name)
    
    print('Selected features: {}'.format(len(selected_features)))
    
    # Filter dataframes to include only selected features
    train_filtered = train[selected_features]
    test_filtered = test[selected_features]
    
    return train_filtered, test_filtered, selected_features, selected_categorical

def get_enhanced_lgb_params():
    """
    Define improved LightGBM parameters
    """
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 7,  # Increased from default
        'learning_rate': 0.01,  # Adjusted for better convergence
        'min_child_samples': 20,
        'subsample': 0.8,  # Added subsample
        'colsample_bytree': 0.8,  # Added column sampling
        'reg_alpha': 0.1,  # Added L1 regularization
        'reg_lambda': 1.0,  # Added L2 regularization
        'min_split_gain': 0.01,
        'min_child_weight': 1,
        'bagging_freq': 1,
        'bagging_fraction': 0.9,
        'bagging_seed': 11,
        'metric': 'f1',
        'nthread': -1,  # Use all available threads
        'verbosity': -1
        # Removed is_unbalance: 'true' - we'll use scale_pos_weight instead
    }
    
    return params
    

def balance_dataset(X, y, method='undersample'):
    """
    Simple class balancing without using SMOTE
    """
    if method == 'undersample':
        # Undersample the majority class
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
        
        # Sample from majority class same number as minority class
        n_minority = len(minority_indices)
        majority_indices_sampled = np.random.choice(majority_indices, size=n_minority, replace=False)
        
        # Combine indices
        balanced_indices = np.concatenate([majority_indices_sampled, minority_indices])
        
        return X.iloc[balanced_indices], y.iloc[balanced_indices]
    
    elif method == 'oversample':
        # Oversample the minority class
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
        
        # Sample from minority class with replacement to match majority class size
        n_majority = len(majority_indices)
        minority_indices_sampled = np.random.choice(minority_indices, size=n_majority, replace=True)
        
        # Combine indices
        balanced_indices = np.concatenate([majority_indices, minority_indices_sampled])
        
        return X.iloc[balanced_indices], y.iloc[balanced_indices]
    
    elif method == 'class_weight':
        # Use class weights instead of resampling
        return X, y
    
    else:
        return X, y

def train_and_predict(train, test, target, features, categorical_feats, n_folds=7):
    """
    Train LightGBM model with KFold cross-validation and make predictions with improvements
    """
    # Initialize stratified KFold for better handling of imbalanced data
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize arrays for out-of-fold predictions and test predictions
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    
    # Initialize dataframe for feature importance
    feature_importance_df = pd.DataFrame()
    
# Get enhanced parameters
    params = get_enhanced_lgb_params()
    
    # Calculate the class distribution for scaling
    pos_scale = np.sum(target == 0) / np.sum(target == 1)
    params['scale_pos_weight'] = pos_scale  # Only use scale_pos_weight, not is_unbalance
    
    # Train and predict per fold
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("Fold n°{}".format(fold_))
        
        X_train, X_val = train.iloc[trn_idx][features], train.iloc[val_idx][features]
        y_train, y_val = target.iloc[trn_idx], target.iloc[val_idx]
        
        # Balance the training data
        X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='oversample')
        
        # Create LightGBM datasets
        trn_data = lgb.Dataset(
            X_train_balanced, 
            label=y_train_balanced, 
            categorical_feature=categorical_feats
        )
        
        val_data = lgb.Dataset(
            X_val, 
            label=y_val, 
            categorical_feature=categorical_feats
        )
        
        # Train model with early stopping
        num_round = 10000
        clf = lgb.train(
            params, 
            trn_data, 
            num_round,
            valid_sets=[trn_data, val_data],
            feval=lgb_f1_score,
            callbacks=[lgb.early_stopping(stopping_rounds=200)]
        )
        
        # Get raw predictions
        val_preds = clf.predict(X_val, num_iteration=clf.best_iteration)
        
        # Find optimal threshold for F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_preds)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        
        print(f"Fold {fold_} - Best threshold: {best_threshold:.4f}")
        
        # Save OOF predictions
        oof[val_idx] = val_preds
        
        # Record feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        # Predict on test set
        test_preds = clf.predict(test[features], num_iteration=clf.best_iteration)
        predictions += test_preds / folds.n_splits
    
    # Calculate overall best threshold from OOF predictions
    precisions, recalls, thresholds = precision_recall_curve(target, oof)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_overall_threshold = thresholds[np.argmax(f1_scores[:-1])]
    
    print(f"Best overall threshold: {best_overall_threshold:.4f}")
    
    # Return the raw predictions, binary predictions, and feature importance
    return oof, predictions, (predictions >= best_overall_threshold).astype(int), feature_importance_df

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
    submission_path = 'sample_submission.csv'  
    output_path_probs = 'enhanced_submission_outliers_likelihood.csv'
    output_path_binary = 'enhanced_submission_outliers_binary.csv'
    
    # Load data
    print('[INFO] Loading train and test data')
    train, test = load_data(train_path, test_path)
    
    # Extract target for future use
    target = train['outliers']
    print('Target distribution:')
    print(target.value_counts(normalize=True))
    
    # Prepare features with enhanced engineering
    print('[INFO] Preparing features with enhanced engineering')
    train_features, test_features, target, features, categorical_feats = prepare_features(train, test, perform_engineering=True)
    
    # Print target distribution
    print('Target values: {}'.format(np.unique(target)))
    
    # Filter features by correlation score
    print('[INFO] Filtering features by correlation score')
    train_filtered, test_filtered, selected_features, selected_categorical = filter_features_by_correlation(
        train_features, test_features, features, categorical_feats, corr_scores_path, threshold=10
    )
    
    # Train and predict with improved model
    print('[INFO] Starting training with 7-fold cross-validation and improved parameters')
    oof, predictions_prob, predictions_binary, feature_importance_df = train_and_predict(
        train_filtered, test_filtered, target, selected_features, selected_categorical, n_folds=7
    )
    
    # Save both probability and binary submissions
    print('[INFO] Creating submission files')
    save_submission(predictions_prob, submission_path, output_path_probs)
    save_submission(predictions_binary, submission_path, output_path_binary)
    
    print('[SUCCESS] Model training and prediction completed')
    print(f'Submissions saved to: {output_path_probs} (probabilities) and {output_path_binary} (binary)')

if __name__ == "__main__":
    main()