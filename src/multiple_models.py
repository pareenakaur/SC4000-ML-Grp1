import pandas as pd
import numpy as np
import os
import gc
import warnings
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configure input/output paths
TRAIN_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\train_features.csv"  # Path to train data with engineered features
TEST_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\test_features.csv"    # Path to test data with engineered features
OUTPUT_PATH = r"C:\Users\Aarushi\Desktop\SC4000\SC4000\output\multiple_model_output.csv"    # Path to save predictions

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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
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
    
    # Convert first_active_month to datetime features if present
    if 'first_active_month' in train.columns:
        try:
            # Handle potential invalid values by setting them to NaT
            train['first_active_month'] = pd.to_datetime(train['first_active_month'], format='%Y-%m', errors='coerce')
            # Replace NaT with the median date
            median_date = train['first_active_month'].dropna().median()
            train['first_active_month'] = train['first_active_month'].fillna(median_date)
            # Create card age feature
            train['card_age_days'] = (pd.Timestamp('2018-02-01') - train['first_active_month']).dt.days
            
            if 'first_active_month' in test.columns:
                # Handle potential invalid values in test set
                test['first_active_month'] = pd.to_datetime(test['first_active_month'], format='%Y-%m', errors='coerce')
                # Fill NaT with the median from training set
                test['first_active_month'] = test['first_active_month'].fillna(median_date)
                test['card_age_days'] = (pd.Timestamp('2018-02-01') - test['first_active_month']).dt.days
        except Exception as e:
            print(f"Warning: Error processing first_active_month: {str(e)}")
            print("Skipping datetime feature generation and dropping first_active_month column")
            if 'first_active_month' in train.columns:
                train = train.drop('first_active_month', axis=1)
            if 'first_active_month' in test.columns:
                test = test.drop('first_active_month', axis=1)
    
    # Memory optimizations
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    
    # Separate target
    target = train['target']
    
    # Drop non-feature columns
    drops = ['target', 'card_id', 'first_active_month']
    train_features = train.drop([col for col in drops if col in train.columns], axis=1)
    test_features = test.drop([col for col in drops if col in test.columns], axis=1)
    
    # Ensure same columns in train and test
    common_cols = list(set(train_features.columns) & set(test_features.columns))
    train_features = train_features[common_cols]
    test_features = test_features[common_cols]
    
    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    
    # Scale features for models that need it
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    return train_features, train_features_scaled, target, test_features, test_features_scaled, test['card_id']

def evaluate_models(X, X_scaled, y, cv=5):
    """
    Evaluate multiple regression models using cross-validation.
    """
    print("\n===== Model Evaluation =====")
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso Regression': Lasso(alpha=0.001, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
        'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }
    
    # Optional: Add SVR if dataset is small enough
    if X.shape[0] < 10000:  # Only use SVR for smaller datasets due to time complexity
        models['SVR'] = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    cv_results = {}
    
    # Use scaled features for these models
    scaled_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']
    
    for name, model in models.items():
        start_time = time()
        
        # Choose appropriate features (scaled or unscaled)
        features = X_scaled if name in scaled_models else X
        
        try:
            # Use negative RMSE as scoring
            cv_scores = cross_val_score(
                model, features, y, 
                cv=cv, 
                scoring='neg_root_mean_squared_error', 
                n_jobs=-1 if name not in ['SVR'] else 1
            )
            
            # Convert negative RMSE to positive
            rmse_scores = -cv_scores
            
            cv_results[name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std(),
                'time': time() - start_time
            }
            
            print(f"{name}: RMSE = {rmse_scores.mean():.6f} (±{rmse_scores.std():.6f}), Time: {time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"{name} failed: {str(e)}")
    
    # Find best model
    best_model = min(cv_results.items(), key=lambda x: x[1]['mean_rmse'])
    print(f"\nBest Model: {best_model[0]} with RMSE = {best_model[1]['mean_rmse']:.6f}")
    
    return cv_results, best_model[0]

def tune_best_model(X, X_scaled, y, best_model_name, n_trials=100, timeout=3600):
    """
    Tune hyperparameters for the best model using Optuna.
    """
    print(f"\n===== Tuning {best_model_name} =====")
    
    # Setup KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Define objective function based on the best model
    if best_model_name == 'LightGBM':
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
            }
            
            rmse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
    
    elif best_model_name == 'XGBoost':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20)
            }
            
            rmse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model = XGBRegressor(**params, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
    
    elif best_model_name == 'Random Forest':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
            
            rmse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
    
    elif best_model_name == 'Gradient Boosting':
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
            
            rmse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
    
    elif best_model_name == 'Extra Trees':
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
            }
            
            rmse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                
                model = ExtraTreesRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
    
    elif best_model_name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        # Use scaled features for these models
        X_use = X_scaled
        
        if best_model_name == 'Ridge Regression':
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0.001, 100, log=True)
                
                rmse_scores = []
                for train_idx, valid_idx in kf.split(X_use):
                    X_train, X_valid = X_use[train_idx], X_use[valid_idx]
                    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                    
                    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_valid)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    rmse_scores.append(rmse)
                
                return np.mean(rmse_scores)
        
        elif best_model_name == 'Lasso Regression':
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0.00001, 1.0, log=True)
                
                rmse_scores = []
                for train_idx, valid_idx in kf.split(X_use):
                    X_train, X_valid = X_use[train_idx], X_use[valid_idx]
                    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                    
                    model = Lasso(alpha=alpha, random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_valid)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    rmse_scores.append(rmse)
                
                return np.mean(rmse_scores)
        
        elif best_model_name == 'ElasticNet':
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0.00001, 1.0, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.01, 0.99)
                
                rmse_scores = []
                for train_idx, valid_idx in kf.split(X_use):
                    X_train, X_valid = X_use[train_idx], X_use[valid_idx]
                    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
                    
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=RANDOM_STATE)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_valid)
                    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
                    rmse_scores.append(rmse)
                
                return np.mean(rmse_scores)
    
    else:
        print(f"No tuning function defined for {best_model_name}. Using default parameters.")
        return None
    
    # Create and run the study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"Best RMSE: {best_rmse:.6f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    return best_params

def train_final_model(X, X_scaled, y, test_X, test_X_scaled, best_model_name, best_params=None):
    """
    Train the final model with the best parameters.
    """
    print(f"\n===== Training Final {best_model_name} Model =====")
    
    # Select appropriate features
    scaled_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'SVR']
    X_train = X_scaled if best_model_name in scaled_models else X
    X_test = test_X_scaled if best_model_name in scaled_models else test_X
    
    # Initialize the model with best parameters if available
    if best_model_name == 'LightGBM' and best_params:
        model = lgb.LGBMRegressor(**best_params, random_state=RANDOM_STATE)
    elif best_model_name == 'XGBoost' and best_params:
        model = XGBRegressor(**best_params, random_state=RANDOM_STATE)
    elif best_model_name == 'Random Forest' and best_params:
        model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif best_model_name == 'Gradient Boosting' and best_params:
        model = GradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    elif best_model_name == 'Extra Trees' and best_params:
        model = ExtraTreesRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    elif best_model_name == 'Ridge Regression' and best_params:
        model = Ridge(alpha=best_params['alpha'], random_state=RANDOM_STATE)
    elif best_model_name == 'Lasso Regression' and best_params:
        model = Lasso(alpha=best_params['alpha'], random_state=RANDOM_STATE)
    elif best_model_name == 'ElasticNet' and best_params:
        model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'], random_state=RANDOM_STATE)
    elif best_model_name == 'Linear Regression':
        model = LinearRegression()
    elif best_model_name == 'SVR':
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    else:
        # Fallback to default model
        print(f"Using default parameters for {best_model_name}")
        if best_model_name == 'LightGBM':
            model = lgb.LGBMRegressor(random_state=RANDOM_STATE)
        elif best_model_name == 'XGBoost':
            model = XGBRegressor(random_state=RANDOM_STATE)
        elif best_model_name == 'Random Forest':
            model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        elif best_model_name == 'Gradient Boosting':
            model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        elif best_model_name == 'Extra Trees':
            model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    # Train on the full dataset
    print("Training final model...")
    model.fit(X_train, y)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Get feature importance if available
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.barh(feature_importance['feature'].head(20), feature_importance['importance'].head(20))
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved to 'feature_importance.png'")
        
        # Save feature importance to file
        feature_importance.to_csv('feature_importance.csv', index=False)
    
    return predictions

def compare_model_performance(cv_results):
    """
    Create a visual comparison of model performance.
    """
    models = list(cv_results.keys())
    rmse_means = [cv_results[model]['mean_rmse'] for model in models]
    rmse_stds = [cv_results[model]['std_rmse'] for model in models]
    times = [cv_results[model]['time'] for model in models]
    
    # Sort by performance
    sorted_indices = np.argsort(rmse_means)
    models = [models[i] for i in sorted_indices]
    rmse_means = [rmse_means[i] for i in sorted_indices]
    rmse_stds = [rmse_stds[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    
    # Create performance comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.barh(models, rmse_means, xerr=rmse_stds, capsize=5)
    plt.xlabel('RMSE (lower is better)')
    plt.title('Model Performance Comparison')
    
    plt.subplot(2, 1, 2)
    plt.barh(models, times)
    plt.xlabel('Time (seconds)')
    plt.title('Execution Time')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("Model comparison plot saved to 'model_comparison.png'")

def main():
    """
    Main function to run the training and prediction pipeline.
    """
    try:
        # Import matplotlib and seaborn
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            print("Seaborn not installed, using default matplotlib style.")
    except ImportError:
        print("Matplotlib not installed. Visualizations will be skipped.")
    
    print("===== Starting Multi-Model Regression =====")
    
    # Prepare data
    train_features, train_features_scaled, target, test_features, test_features_scaled, card_ids = prepare_data()
    
    # Evaluate models
    cv_results, best_model_name = evaluate_models(train_features, train_features_scaled, target)
    
    # Compare model performance
    try:
        compare_model_performance(cv_results)
    except Exception as e:
        print(f"Error creating comparison plot: {str(e)}")
    
    # Tune the best model
    best_params = tune_best_model(train_features, train_features_scaled, target, best_model_name, n_trials=50)
    
    # Train final model
    predictions = train_final_model(
        train_features, train_features_scaled, target, 
        test_features, test_features_scaled, 
        best_model_name, best_params
    )
    
    # Create submission file
    submission = pd.DataFrame({
        'card_id': card_ids,
        'target': predictions
    })
    
    # Save results
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to {OUTPUT_PATH}")
    
    print("\n===== All done! =====")

if __name__ == "__main__":
    main()