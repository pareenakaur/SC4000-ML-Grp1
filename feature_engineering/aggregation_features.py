import pandas as pd
import numpy as np
import gc

def aggregate_transactions(history):
    """
    Aggregate transaction features by card_id.
    """
    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_date': ['min', 'max'],
        'month_lag': ['min', 'max', 'std', 'var'],
        'month_diff': ['mean', 'min', 'max', 'var'],
        'weekend': ['sum', 'mean'],
        'card_id': ['size'],
        'month': ['nunique'],
        'hour': ['nunique'],
        'quarter': ['nunique'],
        'weekofyear': ['nunique'],                
        'dayofweek': ['nunique'],
        'Christmas_Day_2017': ['mean', 'max'],
        'fathers_day_2017': ['mean', 'max'],
        'Children_day_2017': ['mean', 'max'],        
        'Black_Friday_2017': ['mean', 'max'],
        'Valentine_day_2017': ['mean'],
        'Mothers_Day_2017': ['mean'],
        'mode_city_id': ['mean'],
        'mode_merchant_category_id': ['mean'],
        'mode_state_id': ['mean'],
        'mode_subsector_id': ['mean'],
        'mode_month_lag': ['mean'],
    }

    # Group by card_id and aggregate
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    # Add transaction count as a separate column
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    # Merge transaction count with other aggregated features
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

def aggregate_transactions_weekend(history):
    """
    Aggregate transaction features for weekend transactions by card_id.
    """
    agg_func = {
        'category_1': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'mode_city_id': ['mean'],
        'mode_merchant_category_id': ['mean'],
        'mode_state_id': ['mean'],
        'mode_subsector_id': ['mean'],
        'mode_month_lag': ['mean'],
    }

    # Group by card_id and aggregate
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    # Add transaction count as a separate column
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    # Merge transaction count with other aggregated features
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

def aggregate_authorized_transactions(history):
    """
    Compute mean of authorized_flag by card_id.
    """
    agg_fun = {'authorized_flag': ['mean']}
    auth_mean = history.groupby(['card_id']).agg(agg_fun)
    auth_mean.columns = ['auth_mean_'.join(col).strip() for col in auth_mean.columns.values]
    auth_mean.reset_index(inplace=True)
    
    return auth_mean