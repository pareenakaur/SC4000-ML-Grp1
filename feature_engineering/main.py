import pandas as pd
import numpy as np
import gc
import datetime
import warnings
import os

# Import custom modules
from utils import reduce_mem_usage, read_data, load_transaction_data
from transaction_features import (preprocess_transactions, add_time_features, 
                                add_category_aggregations, add_holiday_features,
                                add_mode_features)
from aggregation_features import (aggregate_transactions, aggregate_transactions_weekend,
                                aggregate_authorized_transactions)
from merchant_features import (add_merchant_counts, get_historical_merchant_ids, 
                              get_new_merchant_ids, count_merchant_id)
from date_features import (add_date_difference_features, add_first_active_features,
                          add_ratio_features)
from feature_importance import (get_feature_importances, calculate_null_importances,
                              calculate_feature_scores)

# Configure warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)

def main():
    # Create output directories if they don't exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('data/check_point', exist_ok=True)
    os.makedirs('final/data/check_point', exist_ok=True)
    
    # Load transaction data
    print("Loading transaction data...")
    historical_transactions = load_transaction_data('data/raw/historical_transactions.csv')
    new_transactions = load_transaction_data('data/raw/new_merchant_transactions.csv')
    
    # Create copies for safekeeping
    historical_transactions_copy = historical_transactions.copy(deep=True)
    new_transactions_copy = new_transactions.copy(deep=True)
    
    # Preprocess transactions
    print("Preprocessing transaction data...")
    historical_transactions = preprocess_transactions(historical_transactions)
    new_transactions = preprocess_transactions(new_transactions)
    
    # Add time-based features
    print("Adding time features...")
    historical_transactions = add_time_features(historical_transactions)
    new_transactions = add_time_features(new_transactions)
    
    # Add category aggregations
    print("Adding category aggregations...")
    historical_transactions = add_category_aggregations(historical_transactions)
    new_transactions = add_category_aggregations(new_transactions)
    
    # Add holiday features
    print("Adding holiday features...")
    historical_transactions = add_holiday_features(historical_transactions)
    new_transactions = add_holiday_features(new_transactions)
    
    # Add mode features
    print("Adding mode features...")
    historical_transactions = add_mode_features(historical_transactions)
    new_transactions = add_mode_features(new_transactions)
    
    # Save checkpoint
    print("Saving transaction checkpoints...")
    historical_transactions.to_feather('data/check_point/historical_transactions_checkpoint.feather')
    new_transactions.to_feather('data/check_point/new_transactions_checkpoint.feather')
    
    # Aggregate authorized transactions
    print("Aggregating authorized transactions...")
    auth_mean = aggregate_authorized_transactions(historical_transactions)
    
    # Split historical transactions by authorization flag
    authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
    non_authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]
    
    # Filter weekend transactions
    hist_weekend_transactions = historical_transactions[historical_transactions['weekend'] == 1]
    new_weekend_transactions = new_transactions[new_transactions['weekend'] == 1]
    
    # Load train and test data
    print("Loading train and test data...")
    train = reduce_mem_usage(read_data('data/raw/train.csv'))
    test = reduce_mem_usage(read_data('data/raw/test.csv'))
    
    # Merge auth_mean features
    print("Merging authorized transaction features...")
    train = pd.merge(train, auth_mean, on='card_id', how='left')
    test = pd.merge(test, auth_mean, on='card_id', how='left')
    del auth_mean
    gc.collect()
    
    # Merge non-authorized transaction features
    print("Merging non-authorized transaction features...")
    history = aggregate_transactions(non_authorized_transactions)
    history.columns = ['hist_' + c if c != 'card_id' else c for c in history.columns]
    del non_authorized_transactions
    gc.collect()
    
    train = pd.merge(train, history, on='card_id', how='left')
    test = pd.merge(test, history, on='card_id', how='left')
    del history
    gc.collect()
    
    # Merge authorized transaction features
    print("Merging authorized transaction features...")
    authorized = aggregate_transactions(authorized_transactions)
    authorized.columns = ['auth_' + c if c != 'card_id' else c for c in authorized.columns]
    del authorized_transactions
    gc.collect()
    
    train = pd.merge(train, authorized, on='card_id', how='left')
    test = pd.merge(test, authorized, on='card_id', how='left')
    del authorized
    gc.collect()
    
    # Merge new transaction features
    print("Merging new transaction features...")
    new = aggregate_transactions(new_transactions)
    new.columns = ['new_' + c if c != 'card_id' else c for c in new.columns]
    
    train = pd.merge(train, new, on='card_id', how='left')
    test = pd.merge(test, new, on='card_id', how='left')
    del new
    gc.collect()
    
    # Merge weekend transaction features
    print("Merging weekend transaction features...")
    hist_weekend = aggregate_transactions_weekend(hist_weekend_transactions)
    hist_weekend.columns = ['hist_weekend_' + c if c != 'card_id' else c for c in hist_weekend.columns]
    del hist_weekend_transactions
    gc.collect()
    
    train = pd.merge(train, hist_weekend, on='card_id', how='left')
    test = pd.merge(test, hist_weekend, on='card_id', how='left')
    del hist_weekend
    gc.collect()
    
    new_weekend = aggregate_transactions_weekend(new_weekend_transactions)
    new_weekend.columns = ['new_weekend_' + c if c != 'card_id' else c for c in new_weekend.columns]
    del new_weekend_transactions
    gc.collect()
    
    train = pd.merge(train, new_weekend, on='card_id', how='left')
    test = pd.merge(test, new_weekend, on='card_id', how='left')
    del new_weekend
    gc.collect()
    
    # Add merchant ID features
    print("Adding merchant ID features...")
    # Historical merchant IDs
    hist_merchant_ids = get_historical_merchant_ids()
    for merchant_id in hist_merchant_ids:
        m = count_merchant_id(historical_transactions, merchant_id)
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    # New merchant IDs
    new_merchant_ids = get_new_merchant_ids()
    for merchant_id in new_merchant_ids:
        m = count_merchant_id(new_transactions, merchant_id)
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    # Clean up large datasets no longer needed
    del historical_transactions, new_transactions, m
    gc.collect()
    
    # Save checkpoint
    print("Saving train/test checkpoints...")
    train.to_feather('data/check_point/train_checkpoint.feather')
    test.to_feather('data/check_point/test_checkpoint.feather')
    
    # Prepare for feature engineering
    print("Preparing features for modeling...")
    # Create outlier flag
    train['outliers'] = 0
    train.loc[train['target'] < -30, 'outliers'] = 1
    
    # Save target variable
    outliers = train['outliers']
    target = train['target']
    
    # Add first active features
    print("Adding first active features...")
    train = add_first_active_features(train)
    test = add_first_active_features(test)
    
    # Add date difference features
    print("Adding date difference features...")
    train = add_date_difference_features(train)
    test = add_date_difference_features(test)
    
    # Add ratio features
    print("Adding ratio features...")
    train = add_ratio_features(train)
    test = add_ratio_features(test)
    
    # Fill missing values
    train = train.fillna(0)
    test = test.fillna(0)
    
    # Save processed data
    print("Saving processed data...")
    train.to_csv('final/data/check_point/train_c.csv')
    test.to_csv('final/data/check_point/test_c.csv')
    
    # Define features for modeling
    features = [c for c in train.columns if c not in ['card_id', 'target', 'first_active_month', 'outliers',
                                                     'hist_weekend_purchase_date_min', 'hist_weekend_purchase_date_max',
                                                     'new_weekend_purchase_date_min', 'new_weekend_purchase_date_max']]
    categorical_feats = [c for c in features if 'feature_' in c]
    
    print(f"Number of features: {len(features)}")
    print(f"Number of categorical features: {len(categorical_feats)}")
    
    # Feature Importance Analysis
    print("Calculating actual feature importances...")
    actual_imp_df = get_feature_importances(train=train[features], 
                                          target=target, 
                                          features=features,
                                          categorical_feats=categorical_feats, 
                                          shuffle=False)
    
    # Save actual feature importances
    actual_imp_df.to_csv('final/data/check_point/actual_imp_df.csv')
    
    # Calculate null importances (if needed)
    print("Calculating null feature importances...")
    null_imp_df = calculate_null_importances(train=train[features], 
                                           target=target,
                                           features=features,
                                           categorical_feats=categorical_feats,
                                           nb_runs=80)
    
    # Save null importances
    null_imp_df.to_feather("final/data/check_point/null_imp_df_1.feather")
    
    # Calculate feature scores
    print("Calculating feature correlation scores...")
    correlation_scores_df = calculate_feature_scores(actual_imp_df, null_imp_df)
    
    # Save feature scores
    correlation_scores_df.to_csv('final/output/feature_correlation_main_lgb1.csv', index=False)
    
    print("Feature engineering pipeline completed successfully!")

if __name__ == "__main__":
    main()
    