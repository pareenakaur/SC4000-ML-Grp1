import pandas as pd
import numpy as np
import datetime
import gc
from utils import binarize

def preprocess_transactions(transactions):
    """
    Initial preprocessing of transaction data.
    """
    # Scale purchase amount
    transactions['purchase_amount'] = (transactions['purchase_amount']+0.761920783094309)*66.5423961054881
    
    # Binarize categorical features
    transactions = binarize(transactions)
    
    # Fill NAs for category features
    transactions['category_2'].fillna(1.0, inplace=True)
    transactions['category_3'].fillna('A', inplace=True)
    
    # Convert category_3 to numeric
    transactions['category_3'] = transactions['category_3'].map({'A':0, 'B':1, 'C':2})
    
    # Ensure purchase_date is datetime
    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    
    return transactions

def add_time_features(transactions):
    """
    Add time-based features to the transaction data.
    """
    transactions['year'] = transactions['purchase_date'].dt.year
    transactions['weekofyear'] = transactions['purchase_date'].dt.isocalendar().week
    transactions['month'] = transactions['purchase_date'].dt.month
    transactions['dayofweek'] = transactions['purchase_date'].dt.dayofweek
    transactions['weekend'] = (transactions.purchase_date.dt.weekday >= 5).astype(int)
    transactions['hour'] = transactions['purchase_date'].dt.hour 
    transactions['quarter'] = transactions['purchase_date'].dt.quarter
    transactions['is_month_start'] = transactions['purchase_date'].dt.is_month_start
    
    # Calculate month difference
    transactions['month_diff'] = ((datetime.datetime.today() - transactions['purchase_date']).dt.days) // 30
    transactions['month_diff'] += transactions['month_lag']
    
    # Force category_2 to be float32
    transactions['category_2'] = transactions['category_2'].astype('float32')
    
    return transactions

def add_category_aggregations(transactions):
    """
    Add purchase amount aggregations by category.
    """
    for col in ['category_2', 'category_3']:
        transactions[col+'_mean'] = transactions['purchase_amount'].groupby(transactions[col]).agg('mean')
        transactions[col+'_max'] = transactions['purchase_amount'].groupby(transactions[col]).agg('max')
        transactions[col+'_min'] = transactions['purchase_amount'].groupby(transactions[col]).agg('min')
        transactions[col+'_var'] = transactions['purchase_amount'].groupby(transactions[col]).agg('var')
    
    return transactions

def add_holiday_features(transactions):
    """
    Add features related to key shopping days/holidays.
    """
    # Christmas : December 25 2017
    transactions['Christmas_Day_2017'] = (pd.to_datetime('2017-12-25') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    # Fathers day: August 13 2017
    transactions['fathers_day_2017'] = (pd.to_datetime('2017-08-13') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    # Children's day: October 12 2017
    transactions['Children_day_2017'] = (pd.to_datetime('2017-10-12') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    # Black Friday: November 24 2017
    transactions['Black_Friday_2017'] = (pd.to_datetime('2017-11-24') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    # Mother's Day: May 14 2017
    transactions['Mothers_Day_2017'] = (pd.to_datetime('2017-05-14') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    # Valentine's Day: June 12 2017
    transactions['Valentine_day_2017'] = (pd.to_datetime('2017-06-12') - transactions['purchase_date']).dt.days.apply(
        lambda x: x if x > 0 and x < 100 else 0)
    
    return transactions

def add_mode_features(transactions):
    """
    Add mode features by card_id.
    """
    # Calculate mode for each card_id
    mode_cols = ['city_id', 'merchant_category_id', 'state_id', 'subsector_id', 'month_lag']
    mode_list = transactions.groupby('card_id')[mode_cols].apply(lambda x: x.mode().iloc[0])
    mode_list.columns = ['mode_' + c if c != 'card_id' else c for c in mode_list.columns]
    
    # Merge mode features back to transactions
    transactions = pd.merge(transactions, mode_list, on='card_id', how='left')
    
    # Add purchase_month (duplicate, but kept for consistency with original code)
    transactions['purchase_month'] = transactions['purchase_date'].dt.month
    
    return transactions