import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import time
import gc

def get_feature_importances(train, target, features, categorical_feats=None, shuffle=False, seed=None):
    """
    Calculate feature importances using LightGBM.
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    target : Series
        Target variable
    features : list
        List of feature names
    categorical_feats : list, optional
        List of categorical feature names
    shuffle : bool, default=False
        Whether to shuffle the target (for null importance)
    seed : int, optional
        Random seed for shuffling
        
    Returns:
    --------
    DataFrame
        Feature importances
    """
    y = target.copy()
    if shuffle:
        y = pd.DataFrame(y).sample(frac=1.0, random_state=seed).values.ravel()

    if 'week' in train.columns:
        train['week'] = train['week'].astype('int32')

    # Prepare the dataset
    dtrain = lgb.Dataset(train[features], y, free_raw_data=False, 
                        categorical_feature=categorical_feats)

    # LightGBM parameters
    lgb_params = {
        'device': 'gpu',  # <- USE GPU
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
        'verbose': -1
    }
    
    # Train the model
    clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=3500)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = features
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = mean_squared_error(y, clf.predict(train[features]))

    return imp_df

def calculate_null_importances(train, target, features, categorical_feats=None, nb_runs=80):
    """
    Calculate null importance distribution by shuffling the target.
    
    Parameters:
    -----------
    train : DataFrame
        Training data
    target : Series
        Target variable
    features : list
        List of feature names
    categorical_feats : list, optional
        List of categorical feature names
    nb_runs : int, default=80
        Number of runs with shuffled target
        
    Returns:
    --------
    DataFrame
        Null importance distribution
    """
    null_imp_df = pd.DataFrame()
    start = time.time()
    dsp = ''
    
    for i in range(nb_runs):
        # Get current run importances with shuffled target
        imp_df = get_feature_importances(
            train=train,
            target=target,
            features=features,
            categorical_feats=categorical_feats,
            shuffle=True,
            seed=i
        )
        
        imp_df['run'] = i + 1 
        
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        
        # Display progress
        spent = (time.time() - start) / 60
        dsp = f'Done with {i+1:4d} of {nb_runs:4d} (Spent {spent:5.1f} min)'
        print(f'\r{dsp}', end='')
    
    print()  # New line after progress display
    return null_imp_df

def calculate_feature_scores(actual_imp_df, null_imp_df):
    """
    Calculate feature scores based on comparison with null importance distribution.
    
    Parameters:
    -----------
    actual_imp_df : DataFrame
        Actual feature importances
    null_imp_df : DataFrame
        Null feature importances from multiple runs
        
    Returns:
    --------
    DataFrame
        Feature scores
    """
    correlation_scores = []
    
    for _f in actual_imp_df['feature'].unique():
        # Get null and actual importances for the feature
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps_gain < np.percentile(f_act_imps_gain, 25)).sum() / f_null_imps_gain.size

        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps_split < np.percentile(f_act_imps_split, 25)).sum() / f_null_imps_split.size

        correlation_scores.append((_f, split_score, gain_score))
    
    scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    return scores_df