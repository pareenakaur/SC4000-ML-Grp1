import pandas as pd
import numpy as np
import datetime

def add_date_difference_features(df, first_active_month_col='first_active_month'):
    """
    Add date difference features.
    """
    for prefix in ['hist', 'auth', 'new']:
        # Convert to datetime if not already
        df[f'{prefix}_purchase_date_max'] = pd.to_datetime(df[f'{prefix}_purchase_date_max'])
        df[f'{prefix}_purchase_date_min'] = pd.to_datetime(df[f'{prefix}_purchase_date_min'])
        
        # Date difference between max and min purchase date
        df[f'{prefix}_purchase_date_diff'] = (df[f'{prefix}_purchase_date_max'] - df[f'{prefix}_purchase_date_min']).dt.days
        
        # Average days between purchases
        df[f'{prefix}_purchase_date_average'] = df[f'{prefix}_purchase_date_diff'] / df[f'{prefix}_card_id_size']
        
        # Days from last purchase to today
        df[f'{prefix}_purchase_date_uptonow'] = (datetime.datetime.today() - df[f'{prefix}_purchase_date_max']).dt.days
        
        # Days from first purchase to today
        df[f'{prefix}_purchase_date_uptomin'] = (datetime.datetime.today() - df[f'{prefix}_purchase_date_min']).dt.days
        
        # Days from first active to first purchase
        df[f'{prefix}_first_buy'] = (df[f'{prefix}_purchase_date_min'] - df[first_active_month_col]).dt.days
        
        # Convert datetime to numeric for efficient storage
        for feature in [f'{prefix}_purchase_date_max', f'{prefix}_purchase_date_min']:
            df[feature] = df[feature].astype(np.int64) * 1e-9
    
    return df

def add_first_active_features(df):
    """
    Add features based on first_active_month.
    """
    df["month"] = df["first_active_month"].dt.month
    df["year"] = df["first_active_month"].dt.year
    df['week'] = df["first_active_month"].dt.isocalendar().week
    df['dayofweek'] = df['first_active_month'].dt.dayofweek
    df['first_active_month'] = pd.to_datetime(df["first_active_month"])
    df['days'] = (pd.Timestamp(2018, 2, 1) - df['first_active_month']).dt.days
    
    return df

def add_ratio_features(df):
    """
    Add ratio features between different transaction types.
    """
    # Purchase date differences
    df['diff_purchase_date_diff'] = df['auth_purchase_date_diff'] - df['new_purchase_date_diff'] 
    df['diff_purchase_date_average'] = df['auth_purchase_date_average'] - df['new_purchase_date_average']
    
    # Merchant ratios
    df['hist_00a6ca8a8a_ratio'] = df['hist_M_ID_00a6ca8a8a'] / (df['hist_transactions_count'] + df['auth_transactions_count'])
    df['new_00a6ca8a8a_ratio'] = df['M_ID_00a6ca8a8a'] / df['new_transactions_count']
    
    # Category ratios
    df['category1_auth_ratio'] = df['auth_category_1_sum'] / df['auth_transactions_count']
    df['category_1_new_ratio'] = df['new_category_1_sum'] / df['new_transactions_count']
    
    # Date ratios
    df['date_average_new_auth_ratio'] = df['auth_purchase_date_average'] / df['new_purchase_date_average']
    
    # Holiday ratios
    df['childday_ratio'] = df['auth_Children_day_2017_mean'] / df['new_Children_day_2017_mean']
    df['blackday_ratio'] = df['auth_Black_Friday_2017_mean'] / df['new_Black_Friday_2017_mean']
    df['fatherday_ratio'] = df['auth_fathers_day_2017_mean'] / df['new_fathers_day_2017_mean']
    df['christmasday_ratio'] = df['auth_Christmas_Day_2017_mean'] / df['new_Christmas_Day_2017_mean']
    
    # Time differences
    df['date_uptonow_diff_auth_new'] = df['auth_purchase_date_uptonow'] - df['new_purchase_date_uptonow']
    
    # Weekend ratios
    df['category1_hist_weekend_ratio'] = df['hist_weekend_category_1_sum'] / df['hist_weekend_transactions_count']
    df['category_1_new_weekend_ratio'] = df['new_weekend_category_1_sum'] / df['new_weekend_transactions_count']
    
    return df