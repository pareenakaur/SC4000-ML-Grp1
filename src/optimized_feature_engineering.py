import pandas as pd
import numpy as np
import gc
from datetime import datetime
import holidays

# Memory optimization settings
pd.options.mode.chained_assignment = None  # Turn off SettingWithCopyWarning
pd.set_option('display.max_columns', None)

# Function to optimize dataframe memory usage
def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # For integers
            if pd.api.types.is_integer_dtype(col_type):
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                
            # For floats
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

# Main feature engineering function that processes transactions in chunks
def process_transactions_in_chunks(hist_trans_path, new_trans_path, chunksize=100000):
    """Process transaction data in chunks to avoid memory issues"""
    
    # Define column dtypes to minimize memory usage from the start
    dtypes = {
        'card_id': 'category',
        'merchant_id': 'category',
        'merchant_category_id': 'int16',
        'subsector_id': 'int16',
        'city_id': 'int16',
        'state_id': 'int16',
        'month_lag': 'int8',
        'installments': 'int8',
        'purchase_amount': 'float32',
        'category_1': 'category',
        'category_2': 'category',
        'category_3': 'category'
    }
    
    # Columns to keep (drop the rest)
    columns_to_keep = [
        'card_id', 'merchant_category_id', 'purchase_date', 
        'purchase_amount', 'installments', 'month_lag', 
        'category_1', 'category_2', 'category_3'
    ]
    
    # Initialize Brazilian holidays
    br_holidays = holidays.BR()
    
    # Initialize output dataframes
    card_features = pd.DataFrame()
    
    # Process historical transactions
    print("Processing historical transactions...")
    
    # Function to process each chunk
    def process_chunk(chunk):
        # Drop unnecessary columns
        chunk = chunk[columns_to_keep]
        
        # Convert date
        chunk['purchase_datetime'] = pd.to_datetime(chunk['purchase_date'])
        
        # Extract time features
        chunk['month'] = chunk['purchase_datetime'].dt.month
        chunk['day'] = chunk['purchase_datetime'].dt.day
        chunk['hour'] = chunk['purchase_datetime'].dt.hour
        chunk['dayofweek'] = chunk['purchase_datetime'].dt.dayofweek
        chunk['is_weekend'] = (chunk['dayofweek'] >= 5).astype('int8')
        chunk['is_holiday'] = chunk['purchase_datetime'].dt.date.map(
            lambda x: int(x in br_holidays)
        ).astype('int8')
        
        # Add first and last day of month flags (important for recurring payments)
        chunk['is_month_start'] = (chunk['day'] <= 5).astype('int8')
        chunk['is_month_end'] = (chunk['day'] >= 25).astype('int8')
        
        # Add recency features based on month_lag
        if 'month_lag' in chunk.columns:
            chunk['is_recent'] = (chunk['month_lag'] >= -3).astype('int8')  # Last 3 months
        
        # Convert categorical columns
        if not pd.api.types.is_categorical_dtype(chunk['card_id']):
            chunk['card_id'] = chunk['card_id'].astype('category')
        
        # Optimize memory
        chunk = reduce_mem_usage(chunk)
        
        return chunk
    
    # Initialize dictionaries to store aggregations
    card_aggs = {}
    
    # Process historical transactions
    for chunk in pd.read_csv(hist_trans_path, chunksize=chunksize, usecols=columns_to_keep, dtype=dtypes):
        # Process the chunk
        chunk = process_chunk(chunk)
        
        # Calculate per-card aggregations for this chunk
        for card_id, card_data in chunk.groupby('card_id'):
            # If card not in dict, initialize
            if card_id not in card_aggs:
                card_aggs[card_id] = {
                    'purchase_count': 0,
                    'purchase_amount_sum': 0,
                    'purchase_amount_min': float('inf'),
                    'purchase_amount_max': float('-inf'),
                    'weekend_purchases': 0,
                    'holiday_purchases': 0,
                    'month_nunique': set(),
                    'merchant_nunique': set(),
                    'installments_mean': [],
                    'installments_max': 0
                }
            
            # Update aggregations
            card_aggs[card_id]['purchase_count'] += len(card_data)
            card_aggs[card_id]['purchase_amount_sum'] += card_data['purchase_amount'].sum()
            card_aggs[card_id]['purchase_amount_min'] = min(
                card_aggs[card_id]['purchase_amount_min'], 
                card_data['purchase_amount'].min()
            )
            card_aggs[card_id]['purchase_amount_max'] = max(
                card_aggs[card_id]['purchase_amount_max'], 
                card_data['purchase_amount'].max()
            )
            card_aggs[card_id]['weekend_purchases'] += card_data['is_weekend'].sum()
            card_aggs[card_id]['holiday_purchases'] += card_data['is_holiday'].sum()
            card_aggs[card_id]['month_nunique'].update(card_data['month'].unique())
            card_aggs[card_id]['merchant_nunique'].update(card_data['merchant_category_id'].unique())
            
            # Calculate month_lag stats to capture recency
            if 'month_lag' in card_data.columns:
                recent_txns = card_data[card_data['month_lag'] >= -3]  # Transactions in last 3 months
                if not card_aggs[card_id].get('recent_purchase_count'):
                    card_aggs[card_id]['recent_purchase_count'] = 0
                    card_aggs[card_id]['recent_amount_sum'] = 0
                
                card_aggs[card_id]['recent_purchase_count'] += len(recent_txns)
                card_aggs[card_id]['recent_amount_sum'] += recent_txns['purchase_amount'].sum() if len(recent_txns) > 0 else 0
                
                # Track month lag distribution (temporal pattern)
                if 'month_lag_dist' not in card_aggs[card_id]:
                    card_aggs[card_id]['month_lag_dist'] = {}
                
                for lag, count in card_data['month_lag'].value_counts().items():
                    if lag not in card_aggs[card_id]['month_lag_dist']:
                        card_aggs[card_id]['month_lag_dist'][lag] = 0
                    card_aggs[card_id]['month_lag_dist'][lag] += count
            
            # Calculate seasonal patterns (for Brazil)
            summer_months = [12, 1, 2]  # Southern hemisphere summer
            winter_months = [6, 7, 8]   # Southern hemisphere winter
            
            summer_txns = card_data[card_data['month'].isin(summer_months)]
            winter_txns = card_data[card_data['month'].isin(winter_months)]
            
            if not card_aggs[card_id].get('summer_purchase_count'):
                card_aggs[card_id]['summer_purchase_count'] = 0
                card_aggs[card_id]['summer_amount_sum'] = 0
                card_aggs[card_id]['winter_purchase_count'] = 0
                card_aggs[card_id]['winter_amount_sum'] = 0
            
            card_aggs[card_id]['summer_purchase_count'] += len(summer_txns)
            card_aggs[card_id]['summer_amount_sum'] += summer_txns['purchase_amount'].sum() if len(summer_txns) > 0 else 0
            card_aggs[card_id]['winter_purchase_count'] += len(winter_txns)
            card_aggs[card_id]['winter_amount_sum'] += winter_txns['purchase_amount'].sum() if len(winter_txns) > 0 else 0
            
            # Track day of week patterns
            if 'dow_dist' not in card_aggs[card_id]:
                card_aggs[card_id]['dow_dist'] = {}
            
            for dow, count in card_data['dayofweek'].value_counts().items():
                if dow not in card_aggs[card_id]['dow_dist']:
                    card_aggs[card_id]['dow_dist'][dow] = 0
                card_aggs[card_id]['dow_dist'][dow] += count
            
            # Track hour patterns (time of day)
            if 'hour_dist' not in card_aggs[card_id]:
                card_aggs[card_id]['hour_dist'] = {}
            
            for hour, count in card_data['hour'].value_counts().items():
                if hour not in card_aggs[card_id]['hour_dist']:
                    card_aggs[card_id]['hour_dist'][hour] = 0
                card_aggs[card_id]['hour_dist'][hour] += count
            
            if 'installments' in card_data.columns:
                card_aggs[card_id]['installments_mean'].extend(card_data['installments'].tolist())
                card_aggs[card_id]['installments_max'] = max(
                    card_aggs[card_id]['installments_max'],
                    card_data['installments'].max()
                )
        
        # Force garbage collection
        del chunk
        gc.collect()
    
    # Process new transactions
    print("Processing new transactions...")
    for chunk in pd.read_csv(new_trans_path, chunksize=chunksize, usecols=columns_to_keep, dtype=dtypes):
        # Process the chunk
        chunk = process_chunk(chunk)
        
        # Calculate per-card aggregations for this chunk
        for card_id, card_data in chunk.groupby('card_id'):
            # If card not in dict, initialize
            if card_id not in card_aggs:
                card_aggs[card_id] = {
                    'purchase_count': 0,
                    'purchase_amount_sum': 0,
                    'purchase_amount_min': float('inf'),
                    'purchase_amount_max': float('-inf'),
                    'weekend_purchases': 0,
                    'holiday_purchases': 0,
                    'month_nunique': set(),
                    'merchant_nunique': set(),
                    'installments_mean': [],
                    'installments_max': 0
                }
            
            # Update aggregations
            card_aggs[card_id]['purchase_count'] += len(card_data)
            card_aggs[card_id]['purchase_amount_sum'] += card_data['purchase_amount'].sum()
            card_aggs[card_id]['purchase_amount_min'] = min(
                card_aggs[card_id]['purchase_amount_min'], 
                card_data['purchase_amount'].min()
            )
            card_aggs[card_id]['purchase_amount_max'] = max(
                card_aggs[card_id]['purchase_amount_max'], 
                card_data['purchase_amount'].min()
            )
            card_aggs[card_id]['weekend_purchases'] += card_data['is_weekend'].sum()
            card_aggs[card_id]['holiday_purchases'] += card_data['is_holiday'].sum()
            card_aggs[card_id]['month_nunique'].update(card_data['month'].unique())
            card_aggs[card_id]['merchant_nunique'].update(card_data['merchant_category_id'].unique())
            
            if 'installments' in card_data.columns:
                card_aggs[card_id]['installments_mean'].extend(card_data['installments'].tolist())
                card_aggs[card_id]['installments_max'] = max(
                    card_aggs[card_id]['installments_max'],
                    card_data['installments'].max()
                )
        
        # Force garbage collection
        del chunk
        gc.collect()
    
    # Convert aggregations to dataframe
    print("Creating final feature dataframe...")
    results = []
    for card_id, aggs in card_aggs.items():
        # Calculate derived features
        if aggs['purchase_count'] > 0:
            weekend_ratio = aggs['weekend_purchases'] / aggs['purchase_count']
            holiday_ratio = aggs['holiday_purchases'] / aggs['purchase_count']
            purchase_amount_mean = aggs['purchase_amount_sum'] / aggs['purchase_count']
            purchase_amount_range = aggs['purchase_amount_max'] - aggs['purchase_amount_min']
        else:
            weekend_ratio = 0
            holiday_ratio = 0
            purchase_amount_mean = 0
            purchase_amount_range = 0
            
        # Calculate installments mean if available
        if aggs['installments_mean']:
            installments_mean = sum(aggs['installments_mean']) / len(aggs['installments_mean'])
        else:
            installments_mean = 0
        
        # Calculate temporal features
        # Recent vs historical activity
        recent_purchase_count = aggs.get('recent_purchase_count', 0)
        recent_amount_sum = aggs.get('recent_amount_sum', 0)
        
        if recent_purchase_count > 0 and aggs['purchase_count'] > recent_purchase_count:
            recent_purchase_ratio = recent_purchase_count / aggs['purchase_count']
            recent_amount_ratio = recent_amount_sum / aggs['purchase_amount_sum'] if aggs['purchase_amount_sum'] > 0 else 0
            historical_purchase_count = aggs['purchase_count'] - recent_purchase_count
            recent_avg_amount = recent_amount_sum / recent_purchase_count
            historical_amount_sum = aggs['purchase_amount_sum'] - recent_amount_sum
            historical_avg_amount = historical_amount_sum / historical_purchase_count if historical_purchase_count > 0 else 0
            amount_trend = recent_avg_amount - historical_avg_amount if historical_purchase_count > 0 else 0
        else:
            recent_purchase_ratio = 1 if recent_purchase_count > 0 else 0
            recent_amount_ratio = 1 if recent_amount_sum > 0 else 0
            amount_trend = 0
            
        # Seasonal spending patterns
        summer_purchase_count = aggs.get('summer_purchase_count', 0)
        summer_amount_sum = aggs.get('summer_amount_sum', 0)
        winter_purchase_count = aggs.get('winter_purchase_count', 0)
        winter_amount_sum = aggs.get('winter_amount_sum', 0)
        
        if summer_purchase_count > 0 and winter_purchase_count > 0:
            summer_avg_amount = summer_amount_sum / summer_purchase_count
            winter_avg_amount = winter_amount_sum / winter_purchase_count
            seasonal_amount_ratio = summer_avg_amount / winter_avg_amount if winter_avg_amount > 0 else 0
            seasonal_count_ratio = summer_purchase_count / winter_purchase_count if winter_purchase_count > 0 else 0
        else:
            seasonal_amount_ratio = 0
            seasonal_count_ratio = 0
        
        # Extract temporal distribution features
        month_lag_dist = aggs.get('month_lag_dist', {})
        if month_lag_dist:
            most_common_lag = max(month_lag_dist.items(), key=lambda x: x[1])[0]
            lag_entropy = -sum((count/aggs['purchase_count']) * np.log(count/aggs['purchase_count']) 
                              for count in month_lag_dist.values() if count > 0)
        else:
            most_common_lag = 0
            lag_entropy = 0
            
        # Day of week pattern
        dow_dist = aggs.get('dow_dist', {})
        if dow_dist:
            most_common_dow = max(dow_dist.items(), key=lambda x: x[1])[0]
            weekday_ratio = sum(dow_dist.get(i, 0) for i in range(5)) / sum(dow_dist.values()) if sum(dow_dist.values()) > 0 else 0
        else:
            most_common_dow = 0
            weekday_ratio = 0
            
        # Hour pattern
        hour_dist = aggs.get('hour_dist', {})
        if hour_dist:
            most_common_hour = max(hour_dist.items(), key=lambda x: x[1])[0]
            # Group into time periods
            morning_hours = sum(hour_dist.get(i, 0) for i in range(5, 12))
            afternoon_hours = sum(hour_dist.get(i, 0) for i in range(12, 17))
            evening_hours = sum(hour_dist.get(i, 0) for i in range(17, 21))
            night_hours = sum(hour_dist.get(i, 0) for i in range(21, 24)) + sum(hour_dist.get(i, 0) for i in range(0, 5))
            
            total_hours = sum(hour_dist.values())
            if total_hours > 0:
                morning_ratio = morning_hours / total_hours
                afternoon_ratio = afternoon_hours / total_hours
                evening_ratio = evening_hours / total_hours
                night_ratio = night_hours / total_hours
            else:
                morning_ratio = afternoon_ratio = evening_ratio = night_ratio = 0
        else:
            most_common_hour = 0
            morning_ratio = afternoon_ratio = evening_ratio = night_ratio = 0
        
        # Create result row
        result = {
            'card_id': card_id,
            'purchase_count': aggs['purchase_count'],
            'purchase_amount_sum': aggs['purchase_amount_sum'],
            'purchase_amount_mean': purchase_amount_mean,
            'purchase_amount_min': aggs['purchase_amount_min'] if aggs['purchase_amount_min'] != float('inf') else 0,
            'purchase_amount_max': aggs['purchase_amount_max'] if aggs['purchase_amount_max'] != float('-inf') else 0,
            'purchase_amount_range': purchase_amount_range,
            'weekend_ratio': weekend_ratio,
            'holiday_ratio': holiday_ratio,
            'month_nunique': len(aggs['month_nunique']),
            'merchant_nunique': len(aggs['merchant_nunique']),
            'installments_mean': installments_mean,
            'installments_max': aggs['installments_max'],
            
            # Temporal features
            'recent_purchase_ratio': recent_purchase_ratio,
            'recent_amount_ratio': recent_amount_ratio,
            'amount_trend': amount_trend,
            'seasonal_amount_ratio': seasonal_amount_ratio,
            'seasonal_count_ratio': seasonal_count_ratio,
            'most_common_lag': most_common_lag,
            'lag_entropy': lag_entropy,
            'most_common_dow': most_common_dow,
            'weekday_ratio': weekday_ratio,
            'most_common_hour': most_common_hour,
            'morning_ratio': morning_ratio,
            'afternoon_ratio': afternoon_ratio,
            'evening_ratio': evening_ratio,
            'night_ratio': night_ratio
        }
        
        results.append(result)
    
    # Create final dataframe
    card_features = pd.DataFrame(results)
    
    # Optimize memory usage
    card_features = reduce_mem_usage(card_features)
    
    return card_features

# Main feature engineering function
def run_feature_engineering(train_path, test_path, hist_trans_path, new_trans_path, output_path):
    """Main function to run the feature engineering pipeline"""
    
    print("Starting feature engineering process...")
    
    # Get card-level features from transactions
    card_features = process_transactions_in_chunks(
        hist_trans_path=hist_trans_path,
        new_trans_path=new_trans_path
    )
    
    # Read train and test with optimized dtypes
    train = pd.read_csv(train_path, dtype={'card_id': 'category'})
    test = pd.read_csv(test_path, dtype={'card_id': 'category'})
    
    # Merge features with train and test
    train_features = pd.merge(train, card_features, on='card_id', how='left')
    test_features = pd.merge(test, card_features, on='card_id', how='left')
    
    # Fill NaN values
    train_features.fillna(0, inplace=True)
    test_features.fillna(0, inplace=True)
    
    # Save files
    train_features.to_csv(output_path + 'train_features.csv', index=False)
    test_features.to_csv(output_path + 'test_features.csv', index=False)
    
    print(f"Feature engineering completed. Files saved to {output_path}")
    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    
    return train_features, test_features

# Example usage (commented out)
train_features, test_features = run_feature_engineering(
    train_path=r"C:\Users\Aarushi\Desktop\SC4000\SC4000\data\cleaned_data\train.csv",
    test_path=r"C:\Users\Aarushi\Desktop\SC4000\SC4000\data\cleaned_data\test.csv",
    hist_trans_path=r"C:\Users\Aarushi\Desktop\SC4000\SC4000\data\cleaned_data\cleaned_historical_transactions.csv",
    new_trans_path=r"C:\Users\Aarushi\Desktop\SC4000\SC4000\data\cleaned_data\cleaned_new_merchant_transactions.csv",
    output_path=r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output"
)