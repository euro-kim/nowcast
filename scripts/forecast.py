import numpy as np
import pandas as pd
from .common import init_df, evaluate_forecast, set_seeds, plot_forecast
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense  # Changed GRU to LSTM
import warnings
import sys

def create_sequences(data, lag, input_cols_idx, target_col_idx):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])  # shape: (lag, num_features)
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def linear(seed, horizon, data_file, var0, var1):
    # Set seeds
    set_seeds(seed)
    
    # Load and prepare data
    df = init_df(data_file)

    df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
    df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
    df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
    df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()

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
    df = init_df(data_file)
    df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
    df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
    df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
    df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()

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

# def lstm(seed, horizon, lag, neurons, epochs, batch_size, data_file, var0, var1, optimizer, loss):
#     set_seeds(seed)
#     df = init_df(data_file)

#     # Log-transform and differencing
#     df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
#     df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
#     df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
#     df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()
#     df.dropna(inplace=True)

#     # Add time index
#     df['time_index'] = np.arange(len(df))

#     # Features and target
#     features = [f'diff_log_{var1}', 'time_index']
#     target = f'diff_log_{var0}'
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df[features + [target]])

#     input_cols_idx = [features.index(f) for f in features]
#     target_col_idx = len(features)  # target is the last column

#     # Create sequences
#     X_all, y_all = create_sequences(scaled_data, lag, input_cols_idx, target_col_idx)

#     # Train/test split
#     split_index = len(X_all) - horizon
#     X_train, X_test = X_all[:split_index], X_all[split_index:]
#     y_train, y_test = y_all[:split_index], y_all[split_index:]

#     # Build model
#     model = Sequential([
#         LSTM(neurons, input_shape=(lag, len(input_cols_idx))),
#         Dense(1)
#     ])
#     model.compile(optimizer=optimizer, loss=loss)

#     # Train
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=UserWarning)
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#     # Predict differenced log values
#     y_pred_diff_log = model.predict(X_test).flatten()
#     y_true_diff_log = y_test.flatten()

#     # Reconstruct log values
#     last_log_val = df[f'log_{var0}'].iloc[-horizon - 1]
#     log_pred = np.cumsum(y_pred_diff_log) + last_log_val
#     log_true = np.cumsum(y_true_diff_log) + last_log_val

#     # Back to original scale
#     y_pred = np.exp(log_pred) - 1e-6
#     y_true = np.exp(log_true) - 1e-6

#     # Historical actuals
#     y_past = df[var0].iloc[:-(horizon)]
#     forecast_index = df.index[-horizon:]

#     return y_past, y_true, y_pred, forecast_index

def lstm(seed, horizon, lag, neurons, epochs, batch_size, data_file, var0, var1, optimizer, loss):
    set_seeds(seed)
    df = init_df(data_file)
    df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
    df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
    df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
    df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()
    df.dropna(inplace=True)
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

    model = Sequential([
        LSTM(neurons, input_shape=(lag, len(input_cols_idx))),
        Dense(1)
    ])
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
    y_true = np.exp(log_true) - 1e-6
    y_past = df[var0].iloc[:-(horizon)]
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].dropna().values
    forecast_index = df.index[-horizon:]

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log


# def gru(seed, horizon, lag, neurons, epochs, batch_size, data_file, var0, var1, optimizer, loss):
#     set_seeds(seed)
#     df = init_df(data_file)

#     # Log-transform and difference for stationarity
#     df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
#     df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
#     df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
#     df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()

#     df.dropna(inplace=True)

#     # Add time index
#     df['time_index'] = np.arange(len(df))

#     # Use differenced log values + time index
#     features = [f'diff_log_{var1}', 'time_index']
#     target = f'diff_log_{var0}'

#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df[features + [target]])

#     input_cols_idx = [features.index(f) for f in features]
#     target_col_idx = len(features)  # last column in the scaled data

#     # Sequences
#     X_all, y_all = create_sequences(scaled_data, lag, input_cols_idx, target_col_idx)

#     split_index = len(X_all) - horizon
#     X_train, X_test = X_all[:split_index], X_all[split_index:]
#     y_train, y_test = y_all[:split_index], y_all[split_index:]

#     # Model
#     model = Sequential([
#         GRU(neurons, input_shape=(lag, len(input_cols_idx))),
#         Dense(1)
#     ])
#     model.compile(optimizer=optimizer, loss=loss)

#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=UserWarning)
#         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

#     # Predict differenced log values
#     y_pred_diff_log = model.predict(X_test).flatten()
#     y_true_diff_log = y_test.flatten()

#     # Reconstruct log values from last known log value
#     last_log_val = df[f'log_{var0}'].iloc[-horizon - 1]
#     log_pred = np.cumsum(y_pred_diff_log) + last_log_val
#     log_true = np.cumsum(y_true_diff_log) + last_log_val

#     # Reconstruct original scale
#     y_pred = np.exp(log_pred) - 1e-6
#     y_true = np.exp(log_true) - 1e-6

#     y_past = df[var0].iloc[:-(horizon)]
#     forecast_index = df.index[-horizon:]

#     return y_past, y_true, y_pred, forecast_index

def gru(seed, horizon, lag, neurons, epochs, batch_size, data_file, var0, var1, optimizer, loss):
    set_seeds(seed)
    df = init_df(data_file)
    df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
    df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
    df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
    df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()
    df.dropna(inplace=True)
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

    model = Sequential([
        GRU(neurons, input_shape=(lag, len(input_cols_idx))),
        Dense(1)
    ])
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
    y_true = np.exp(log_true) - 1e-6
    y_past = df[var0].iloc[:-(horizon)]
    y_past_diff_log = df[f'diff_log_{var0}'].iloc[:-(horizon)].dropna().values
    forecast_index = df.index[-horizon:]

    return y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log
