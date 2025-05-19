import os, warnings, sys
import numpy as np
from numpy import array
import pandas as pd
from pandas import DataFrame, Series
from .common import ResponseVariable, init_df, diff_df, set_seeds
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense  

def create_sequences(data, lag, input_cols_idx, target_col_idx): 
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])  # shape: (lag, num_features)
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def arima(seed, maxlags, horizon, data_file, var0) -> ResponseVariable:
    np.random.seed(seed)
    
    # Load and preprocess data
    df : DataFrame = init_df(data_file)
    df : DataFrame = diff_df(df, var0)

    # Result initialization
    result = ResponseVariable()
    result.title = f'ARIMA {var0}'  
    result.forecast_index = df.index[-horizon:]

    # Define Series
    series : Series = df[var0]
    log_series : Series = df[f'log_{var0}']
    diff_log_series : Series = df[f'diff_log_{var0}']

    # Actual values    
    result.past = series.iloc[:-(horizon)]
    result.true = series.iloc[-horizon:]

    # Actual diff log values
    result.diff_log_past = diff_log_series.iloc[:-(horizon)]
    result.diff_log_true = diff_log_series.iloc[-horizon:]


    train_data : Series = diff_log_series[:-horizon] # same as ## series.iloc[:-(horizon)] 

    if len(train_data) < maxlags:
        print(f"Error: Not enough training data. len(train_data)={len(train_data)}, maxlags={maxlags}")
        sys.exit(1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = ARIMA(train_data, order=(maxlags, 0, 0))
            fitted_model = model.fit()
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        sys.exit(1)

    # Forecast in differenced log space
    result.diff_log_pred = fitted_model.forecast(steps=horizon)

    # Convert forecast back to original scale
    last_log : Series = log_series.iloc[-horizon - 1]
    forecast_log : Series = result.diff_log_pred.cumsum() + last_log
    result.pred  = np.exp(forecast_log) - 1e-6

    return result


def garch(seed, p, q, horizon, data_file, var0) -> ResponseVariable:
    np.random.seed(seed)
    
    # Load and preprocess data
    df = init_df(data_file)
    df = diff_df(df, var0)

    # Define Series
    series : Series = df[var0]
    log_series : Series = df[f'log_{var0}']
    diff_log_series : Series = df[f'diff_log_{var0}']

    # Result initialization
    result = ResponseVariable()
    result.title = f'GARCH {var0}'  
    result.forecast_index = df.index[-horizon:]

    # Actual values    
    result.past = series.iloc[:-(horizon)]
    result.true = series.iloc[-horizon:]

    # Actual diff log values
    result.diff_log_past = diff_log_series.iloc[:-(horizon)]
    result.diff_log_true = diff_log_series.iloc[-horizon:]


    train_data : Series = diff_log_series[:-horizon]

    if len(train_data) < max(p, q) + 1:
        print(f"Error: Not enough training data. len(train_data)={len(train_data)}, required={max(p, q) + 1}")
        sys.exit(1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = arch_model(train_data, p=p, q=q, vol='GARCH', dist='normal')
            fitted_model = model.fit(disp='off')
    except Exception as e:
        print(f"Error fitting GARCH model: {e}")
        sys.exit(1)

    # Forecast in differenced log space
    forecast_result = fitted_model.forecast(horizon=horizon)
    result.diff_log_pred = forecast_result.mean.iloc[-1]

    # Convert forecast back to original scale
    last_log : Series = log_series.iloc[-horizon - 1]
    forecast_log : Series = result.diff_log_pred.cumsum() + last_log
    result.pred  = np.exp(forecast_log) - 1e-6

    return result

def linear(seed, horizon, data_file, var0, var1):
    # Set seeds
    set_seeds(seed)

    # Load and prepare data
    df = init_df(data_file)
    df = diff_df(df, var0, var1)

    # Feature (X) and target (y)
    df_stationary = df[[f'diff_log_{var0}', f'diff_log_{var1}']].dropna()

    X = df_stationary[[f'diff_log_{var1}']].values
    y = df_stationary[f'diff_log_{var0}'].values

    # Split train/test
    train_X = X[:-horizon]
    test_X = X[-horizon:]
    train_y = y[:-horizon]
    y_true_diff_log = y[-horizon:]

    # Linear Regression
    model = LinearRegression()
    model.fit(train_X, train_y)
    y_pred_diff_log = model.predict(test_X)

    last_log_val = df[f'log_{var0}'].iloc[-horizon - 1]
    log_pred = np.cumsum(y_pred_diff_log) + last_log_val
    log_true = np.cumsum(y_true_diff_log) + last_log_val

    y_pred = np.exp(log_pred) - 1e-6
    y_true = np.exp(log_true) - 1e-6

    # Get actual values of var0 from the past (to display)
    y_past = df[var0].iloc[:-(horizon)]
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].values
    # Get index
    forecast_index = df.index[-horizon:]
    

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log

def var(seed, maxlags, horizon, data_file, var0, var1, ic):
    # Set seeds
    set_seeds(seed)
    df = init_df(data_file)
    df = diff_df(df, var0, var1)

    diff_log_features = [f'diff_log_{var0}', f'diff_log_{var1}']
    model_data = df[diff_log_features].dropna()
    train_data = model_data[:-horizon]
    test_data = model_data[-horizon:]

    if len(train_data) < maxlags:
        print(f"Error: Not enough training data.  len(train_data)={len(train_data)}, maxlags={maxlags}")
        sys.exit(1)

    try:
        model = VAR(train_data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            regression_results = model.fit(maxlags=maxlags, ic=ic)
    except Exception as e:
        print(f"Error fitting VAR model: {e}")
        sys.exit(1)

    lag_order = regression_results.k_ar
    forecast_input = train_data.values[-lag_order:]
    forecast_diff_log = regression_results.forecast(y=forecast_input, steps=horizon)
    forecast_df = pd.DataFrame(forecast_diff_log, columns=diff_log_features)

    last_log_var0 = df[f'log_{var0}'].iloc[-horizon - 1]
    forecast_df[f'log_{var0}'] = forecast_df[f'diff_log_{var0}'].cumsum() + last_log_var0
    forecast_df[f'{var0}_forecast'] = np.exp(forecast_df[f'log_{var0}']) - 1e-6

    y_true = df[var0].iloc[-horizon:]
    y_past = df[var0].iloc[:-(horizon)]
    y_true_diff_log = df[f'diff_log_{var0}'].iloc[-horizon:].values
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].dropna().values
    y_pred_diff_log = forecast_df[f'diff_log_{var0}'].values

    forecast_index = df.index[-horizon:]
    y_pred = forecast_df[f'{var0}_forecast'].values
    

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log

def lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss):
    set_seeds(seed)
    df = init_df(data_file)
    df = diff_df(df, var0, var1)

    df['time_index'] = np.arange(len(df))

    features = [f'diff_log_{var1}', 'time_index']
    target = f'diff_log_{var0}'
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features + [target]])

    input_cols_idx = [features.index(f) for f in features]
    target_col_idx = len(features)

    X_all, y_all = create_sequences(scaled_data, lag, input_cols_idx, target_col_idx)

    split_index = len(X_all) - horizon
    X_train, X_test = X_all[:split_index], X_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    model = Sequential()
    for i in range(layers):
        return_sequences = i < layers - 1  # Only the last LSTM layer should return a single output
        if i == 0:
            model.add(LSTM(neurons, return_sequences=return_sequences, input_shape=(lag, len(input_cols_idx))))
        else:
            model.add(LSTM(neurons, return_sequences=return_sequences))

    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    y_pred_diff_log = model.predict(X_test).flatten()
    y_true_diff_log = y_test.flatten()

    last_log_val = df[f'log_{var0}'].iloc[-horizon - 1]
    log_pred = np.cumsum(y_pred_diff_log) + last_log_val
    log_true = np.cumsum(y_true_diff_log) + last_log_val

    y_true = df[var0].iloc[-horizon:]
    y_past = df[var0].iloc[:-(horizon)]
    y_pred = np.exp(log_pred) - 1e-6
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].dropna().values
    forecast_index = df.index[-horizon:]
    

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log


def gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss):
    set_seeds(seed)
    df = init_df(data_file)
    df = diff_df(df, var0, var1)

    df['time_index'] = np.arange(len(df))

    features = [f'diff_log_{var1}', 'time_index']
    target = f'diff_log_{var0}'
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features + [target]])

    input_cols_idx = [features.index(f) for f in features]
    target_col_idx = len(features)

    X_all, y_all = create_sequences(scaled_data, lag, input_cols_idx, target_col_idx)

    split_index = len(X_all) - horizon
    X_train, X_test = X_all[:split_index], X_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    model = Sequential()
    for i in range(layers):
        return_sequences = i < layers - 1
        if i == 0:
            model.add(GRU(neurons, return_sequences=return_sequences, input_shape=(lag, len(input_cols_idx))))
        else:
            model.add(GRU(neurons, return_sequences=return_sequences))

    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    y_pred_diff_log = model.predict(X_test).flatten()
    y_true_diff_log = y_test.flatten()

    last_log_val = df[f'log_{var0}'].iloc[-horizon - 1]
    log_pred = np.cumsum(y_pred_diff_log) + last_log_val
    log_true = np.cumsum(y_true_diff_log) + last_log_val

    y_pred = np.exp(log_pred) - 1e-6
    y_true = df[var0].iloc[-horizon:]
    y_past = df[var0].iloc[:-(horizon)]
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].dropna().values
    forecast_index = df.index[-horizon:]
    

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log
