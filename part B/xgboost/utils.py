import numpy as np
import pandas as pd
import cupy as cp
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# load data and set random seed
df = pd.read_csv('../data/exchange_rate_imputed.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
random_seed = 42

def create_lags(df, lags,columns=None):
    """Create lagged features for time series forecasting.

    Args:
        df (_type_): Dataframe containing the time series data.
        lags (_type_): Number of lagged features to create.
        columns (_type_, optional): Columns to create lagged features for. If None, all columns are used. Default is None.

    Returns:
        _type_: Dataframe containing the original and lagged features.
    """
    if columns is None:
        columns = df.columns
        
    df = df.copy()  
    for lag in range(lags):
        lag = lag + 1
        for column in columns:       
            df[f'{column}_lag_{lag}'] = df[column].shift(lag).copy()
    return df

def create_future(df, future, columns=None):
    """ Create future values for time series forecasting.

    Args:
        df (_type_): Dataframe containing the time series data.
        future (_type_): Number of future steps to predict.
        columns (_type_, optional): Columns to create future values for. If None, all columns are used. Default is None.

    Returns:
        _type_: Dataframe containing the original and future values.
    """
    if columns is None:
        columns = df.columns
        
    df = df.copy()
    for column in columns:
        for step in range(future):
            step = step + 1
            df[f'{column}_future_{step}'] = df[column].shift(-step).copy()
    return df

def create_data(df, lags, steps, x_columns=None, y_columns=None, test_size=100, cuda=False, verbose=False):
    """
    Create training and testing data for time series forecasting.

    Args:
        df (pandas.DataFrame): The input dataframe containing the time series data.
        lags (int): The number of lagged features to create.
        steps (int): The number of future steps to predict.
        x_columns (list, optional): The columns to use as input features. If None, all columns are used. Default is None.
        y_columns (list, optional): The columns to use as target variables. If None, all columns except x_columns are used. Default is None.
        test_size (int, optional): The number of samples to use for testing. Default is 100.
        cuda (bool, optional): Whether to move data into GPU. Default is False.
        verbose (bool, optional): Whether to print verbose output. Default is False.

    Returns:
        tuple: A tuple containing the training and testing data in the following order: (X_train, X_test, y_train, y_test).
    """

    cols = set(df.columns.to_list()).difference(set(y_columns))
    
    if x_columns is None:
        x_columns = df.columns
    
    df1 = create_lags(df, lags, x_columns) 
    df1 = create_future(df1, steps, columns=y_columns)
    df1.dropna(inplace=True)
    
    columns_x = [col for col in df1.columns if 'lag' in col]
    columns_y = [col for col in df1.columns if col not in columns_x] 
    columns_y = [col for col in columns_y if col not in cols]
    X = df1[columns_x]
    y = df1[columns_y]
        
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    X_train = cp.array(X_train.values).astype(cp.float32)
    X_test = cp.array(X_test.values).astype(cp.float32)
    y_train = cp.array(y_train.values).astype(cp.float32)
    y_test = cp.array(y_test.values).astype(cp.float32)
    
    if cuda:
        X_train = X_train.to_gpu()
        X_test = X_test.to_gpu()
        y_train = y_train.to_gpu()
        y_test = y_test.to_gpu()
    
    if verbose:
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def objective(trial):
    """ Objective function for Optuna optimization.

    Args:
        trial (_type_): The Optuna trial object.

    Raises:
        optuna.TrialPruned: If the trial is pruned.

    Returns:
        _type_: The mean absolute error of the model.
    """
    
    # Lag hyperparameter
    lag = trial.suggest_int("lag", 30, 400, step=50)
    
    # Define hyperparameters for XGBoost 
    max_depth = trial.suggest_int("max_depth", 1, 20)
    n_estimators = trial.suggest_int("n_estimators", 50, 1200, step=50)
    gamma = trial.suggest_float("gamma", 0.1, 1.0, step=0.1)
    subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1)
    reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 1e-1, log=True)
    reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    
    # Get training and testing dat
    X_train, X_val, y_train, y_val = create_data(df, lags=lag, steps=100, y_columns=['6'], test_size=200, cuda=True)
        
    # Create and train the model
    xgb = XGBRegressor(tree_method="hist", random_state=random_seed, device='cuda', eval_metric = 'mae',
                       max_depth=max_depth, n_estimators=n_estimators, gamma=gamma, subsample=subsample,
                       colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                       learning_rate=learning_rate)
    
    xgb.fit(X_train, y_train)
       
    # Evaluate the model
    y_pred = xgb.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    
    trial.report(mae)
    
    # Check for pruning
    if trial.should_prune():
        raise optuna.TrialPruned()

    return mae