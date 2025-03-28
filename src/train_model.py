import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
import os
import warnings
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Optuna settings
N_TRIALS = 100  # Number of parameter search trials
TIMEOUT = 3600  # Maximum time for optimization in seconds (1 hour)

# Configure input/output paths
# Configure input/output paths
TRAIN_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\train_features.csv"  # Path to train data with engineered features
TEST_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\test_features.csv"    # Path to test data with engineered features
OUTPUT_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\run_1\output.csv"    # Path to save predictions

def reduce_mem_usage(df):
    """
    Reduce memory usage of a dataframe by using more efficient data types.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category' and 'datetime' not in str(col_type):
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
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)  # Using float32 as float16 can be unstable
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

def prepare_data():
    """
    Load and prepare data for training.
    """
    print("Loading data...")
    
    # Load train and test data
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    # Memory optimizations
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    
    # Separate target
    target = train['target']
    
    # Drop non-feature columns
    drops = ['target', 'card_id', 'first_active_month']
    train_features = train.drop(drops, axis=1)
    test_features = test.drop([col for col in drops if col in test.columns], axis=1)
    
    # Ensure same columns in train and test
    common_cols = list(set(train_features.columns) & set(test_features.columns))
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]
    
    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    
    return train_features, target, test_features, test['card_id']

def objective(trial, train_features, target, folds):
    """
    Optuna objective function for hyperparameter optimization.
    """
    # Define hyperparameters to optimize
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        
        # Hyperparameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
    }
    
    # Cross-validation scores
    cv_scores = []
    fold_predictions = np.zeros(len(train_features))
    
    # Perform k-fold cross validation
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(train_features)):
        X_train, X_valid = train_features.iloc[train_idx], train_features.iloc[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
        
        # Create dataset for lightgbm
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=params['n_estimators'],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
        )
        
        # Make predictions
        preds = model.predict(X_valid)
        fold_predictions[valid_idx] = preds
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        cv_scores.append(rmse)
        
        # Clean up to free memory
        del X_train, X_valid, y_train, y_valid, train_data, valid_data, model, preds
        gc.collect()
    
    # Return mean CV score
    mean_rmse = np.mean(cv_scores)
    print(f"Trial completed with mean RMSE: {mean_rmse:.6f}")
    
    return mean_rmse

def train_optimal_model(best_params, train_features, target, test_features):
    """
    Train the final model using the optimal hyperparameters.
    """
    # Setup KFold
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Storage for oof and test predictions
    oof_preds = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features))
    feature_importance_df = pd.DataFrame()
    
    # Perform k-fold cross validation with the optimal parameters
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(train_features)):
        print(f'Training fold {fold_n + 1}')
        
        X_train, X_valid = train_features.iloc[train_idx], train_features.iloc[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
        
        # Create dataset for lightgbm
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        
        # Train model
        model = lgb.train(
            best_params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=best_params['n_estimators'],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
        )
        
        # Feature importance
        fold_importance = pd.DataFrame()
        fold_importance['feature'] = train_features.columns
        fold_importance['importance'] = model.feature_importance(importance_type='gain')
        fold_importance['fold'] = fold_n + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
        
        # Make predictions
        oof_preds[valid_idx] = model.predict(X_valid)
        test_preds += model.predict(test_features) / folds.n_splits
        
        # Clean up
        del X_train, X_valid, y_train, y_valid, train_data, valid_data, model
        gc.collect()
    
    # Overall validation score
    overall_rmse = np.sqrt(mean_squared_error(target, oof_preds))
    print(f'Overall RMSE: {overall_rmse:.6f}')
    
    # Plot feature importance
    plot_feature_importance(feature_importance_df)
    
    return test_preds, oof_preds, overall_rmse, feature_importance_df

def plot_feature_importance(feature_importance_df):
    """
    Plot the feature importance from the trained model.
    """
    # Display feature importance
    cols = feature_importance_df[['feature', 'importance']].groupby('feature').mean().sort_values(
        by='importance', ascending=False)[:20].index
    
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

def main():
    """
    Main function to run the training and prediction pipeline.
    """
    try:
        # Import seaborn if available (for nice plots)
        global sns
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Feature importance plot will use matplotlib only.")
        global sns
        class DummySNS:
            @staticmethod
            def barplot(*args, **kwargs):
                data = kwargs.get('data')
                x = kwargs.get('x')
                y = kwargs.get('y')
                sorted_data = data.sort_values(by=x, ascending=False)
                plt.barh(sorted_data[y], sorted_data[x])
        sns = DummySNS()
    
    print("Starting model training process...")
    
    # Prepare data
    train_features, target, test_features, card_ids = prepare_data()
    
    # Set up KFold for cross-validation
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Hyperparameter optimization with Optuna
    print("\nStarting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_features, target, folds),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    print(f"\nBest RMSE: {best_value:.6f}")
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Train the model with the best hyperparameters
    print("\nTraining final model with optimal hyperparameters...")
    test_preds, oof_preds, overall_rmse, feature_importance_df = train_optimal_model(
        best_params, train_features, target, test_features
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'card_id': card_ids,
        'target': test_preds
    })
    
    # Save results
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    
    # Save feature importance
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to feature_importance.csv and feature_importance.png")
    
    print("\nAll done!")

if __name__ == "__main__":
    main()