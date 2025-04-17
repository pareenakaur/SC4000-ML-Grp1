import pandas as pd
import numpy as np

def count_merchant_id(transactions, merchant_id, prefix=''):
    """
    Count occurrences of a specific merchant_id per card_id.
    """
    m = transactions.groupby(['card_id'])['merchant_id'].apply(lambda x: x[x==merchant_id].count())
    m = m.to_frame()
    m = m.rename(columns={'merchant_id': f'{prefix}{merchant_id}'})
    return m

def add_merchant_counts(train, test, transactions, merchant_ids, prefix=''):
    """
    Add counts of specific merchant_ids to train and test sets.
    """
    for merchant_id in merchant_ids:
        m = count_merchant_id(transactions, merchant_id, prefix)
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    return train, test

def get_historical_merchant_ids():
    """
    Return list of historical merchant IDs to track.
    """
    return [
        'M_ID_57df19bf28', 'M_ID_19171c737a', 'M_ID_8fadd601d2', 'M_ID_9e84cda3b1',
        'M_ID_b794b9d9e8', 'M_ID_ec24d672a3', 'M_ID_5a0a412718', 'M_ID_490f186c5a',
        'M_ID_77e2942cd8', 'M_ID_fc7d7969c3', 'M_ID_0a00fa9e8a', 'M_ID_1f4773aa76',
        'M_ID_5ba019a379', 'M_ID_ae9fe1605a', 'M_ID_48257bb851', 'M_ID_1d8085cf5d',
        'M_ID_820c7b73c8', 'M_ID_1ceca881f0', 'M_ID_00a6ca8a8a', 'M_ID_e5374dabc0',
        'M_ID_9139332ccc', 'M_ID_5d4027918d', 'M_ID_9fa00da7b2', 'M_ID_d7f0a89a87',
        'M_ID_daeb0fe461', 'M_ID_077bbb4469', 'M_ID_f28259cb0a', 'M_ID_0a767b8200',
        'M_ID_fee47269cb', 'M_ID_c240e33141', 'M_ID_a39e6f1119', 'M_ID_c8911208f2',
        'M_ID_9a06a8cf31', 'M_ID_08fdba20dc', 'M_ID_a483a17d19', 'M_ID_aed77085ce',
        'M_ID_25d0d2501a', 'M_ID_69f024d01a'
    ]

def get_new_merchant_ids():
    """
    Return list of new merchant IDs to track.
    """
    return [
        'M_ID_6f274b9340', 'M_ID_445742726b', 'M_ID_4e461f7e14', 'M_ID_e5374dabc0',
        'M_ID_3111c6df35', 'M_ID_50f575c681', 'M_ID_00a6ca8a8a', 'M_ID_cd2c0b07e9',
        'M_ID_9139332ccc'
    ]

def process_historical_merchant_ids(historical_transactions, train, test):
    """
    Process historical merchant IDs using the exact same approach as the original code.
    """
    # List of merchant IDs with normal naming
    regular_merchants = [
        'M_ID_57df19bf28', 'M_ID_19171c737a', 'M_ID_8fadd601d2', 'M_ID_9e84cda3b1',
        'M_ID_b794b9d9e8', 'M_ID_ec24d672a3', 'M_ID_5a0a412718', 'M_ID_490f186c5a',
        'M_ID_77e2942cd8', 'M_ID_fc7d7969c3', 'M_ID_0a00fa9e8a', 'M_ID_1f4773aa76',
        'M_ID_5ba019a379', 'M_ID_ae9fe1605a', 'M_ID_48257bb851', 'M_ID_1d8085cf5d',
        'M_ID_820c7b73c8', 'M_ID_1ceca881f0', 'M_ID_5d4027918d', 'M_ID_9fa00da7b2',
        'M_ID_d7f0a89a87', 'M_ID_daeb0fe461', 'M_ID_077bbb4469', 'M_ID_f28259cb0a',
        'M_ID_0a767b8200', 'M_ID_fee47269cb', 'M_ID_c240e33141', 'M_ID_a39e6f1119',
        'M_ID_c8911208f2', 'M_ID_9a06a8cf31', 'M_ID_08fdba20dc', 'M_ID_a483a17d19',
        'M_ID_aed77085ce', 'M_ID_25d0d2501a', 'M_ID_69f024d01a'
    ]
    
    # List of merchant IDs that need "hist_" prefix
    hist_prefix_merchants = [
        'M_ID_00a6ca8a8a', 'M_ID_e5374dabc0', 'M_ID_9139332ccc'
    ]
    
    # Process regular merchants
    for merchant_id in regular_merchants:
        m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(
            lambda x: x[x==merchant_id].count())
        m = m.to_frame()
        m = m.rename(columns={'merchant_id': merchant_id})
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    # Process merchants with hist_ prefix
    for merchant_id in hist_prefix_merchants:
        m = historical_transactions.groupby(['card_id'])['merchant_id'].apply(
            lambda x: x[x==merchant_id].count())
        m = m.to_frame()
        m = m.rename(columns={'merchant_id': f'hist_{merchant_id}'})
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    return train, test

def process_new_merchant_ids(new_transactions, train, test):
    """
    Process new merchant IDs using the exact same approach as the original code.
    """
    # List of new merchant IDs
    new_merchants = [
        'M_ID_6f274b9340', 'M_ID_445742726b', 'M_ID_4e461f7e14', 'M_ID_e5374dabc0',
        'M_ID_3111c6df35', 'M_ID_50f575c681', 'M_ID_00a6ca8a8a', 'M_ID_cd2c0b07e9',
        'M_ID_9139332ccc'
    ]
    
    # Process new merchants
    for merchant_id in new_merchants:
        m = new_transactions.groupby(['card_id'])['merchant_id'].apply(
            lambda x: x[x==merchant_id].count())
        m = m.to_frame()
        m = m.rename(columns={'merchant_id': merchant_id})
        train = pd.merge(train, m, on='card_id', how='left')
        test = pd.merge(test, m, on='card_id', how='left')
    
    return train, test