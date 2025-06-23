import os, warnings, sys
# Suppress TensorFlow and oneDNN info/warning messages globally
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN custom op info messages

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from scripts.models.response_variable import ResponseVariable
from scripts.common.seed import set_seeds
from scripts.common.structure import init_df, diff_df
import pandas as pd
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def lstm(seed: int, horizon: int, lag: int, neurons: int, layers: int, epochs: int, batch_size: int, data_file: str, vars: list[str], optimizer: str, loss: str) -> ResponseVariable:
    set_seeds(seed)
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'LSTM'
    result.model_name = f'LSTM({seed}%{neurons}%{layers}%{epochs}%{batch_size})'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    df['time_index'] = np.arange(len(df))
    features = [f'diff_log_{v}' for v in vars[1:]] + ['time_index']
    scaler = MinMaxScaler()
    for var in vars:
        
        target = f'diff_log_{var}'
        data_for_model = df[features + [target]].dropna()
        scaled_data = scaler.fit_transform(data_for_model)
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
                model.add(LSTM(neurons, return_sequences=return_sequences, input_shape=(lag, len(input_cols_idx))))
            else:
                model.add(LSTM(neurons, return_sequences=return_sequences))
        model.add(Dense(1))
        model.compile(optimizer=optimizer, loss=loss)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred_scaled = model.predict(X_test)
        dummy_array = np.zeros((len(y_pred_scaled), len(features) + 1))
        dummy_array[:, -1] = y_pred_scaled.flatten()
        y_pred_original_scale = scaler.inverse_transform(dummy_array)[:, -1]
        forecast_index = df.index[-horizon:]
        result.diff_log_pred[var] = pd.Series(y_pred_original_scale, index=forecast_index)
        log_series = df[f'log_{var}']
        last_log = log_series.iloc[-horizon - 1]
        log_pred = np.cumsum(result.diff_log_pred[var]) + last_log
        result.pred[var] = np.exp(log_pred) - 1e-6

        series = df[var]
        diff_log_series = df[f'diff_log_{var}']
        result.set_series_segments(var, series, diff_log_series, horizon)

        # Save fitted model and parameters
        result.fitted[var] = model
        # ...no parameter saving for RNN models...
        result.save_params_csv(var)
        # Call Diebold-Mariano test for each variable (LSTM)
        result.diebold_mariano_test(var)

    return result

def gru(seed: int, horizon: int, lag: int, neurons: int, layers: int, epochs: int, batch_size: int, data_file: str, vars: list[str], optimizer: str, loss: str) -> ResponseVariable:
    set_seeds(seed)
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'GRU'
    result.model_name = f'GRU({seed}%{neurons}%{layers}%{epochs}%{batch_size})'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    df['time_index'] = np.arange(len(df))
    features = [f'diff_log_{v}' for v in vars[1:]] + ['time_index']
    scaler = MinMaxScaler()
    for var in vars:
        
        target = f'diff_log_{var}'
        data_for_model = df[features + [target]].dropna()
        scaled_data = scaler.fit_transform(data_for_model)
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
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred_scaled = model.predict(X_test)
        dummy_array = np.zeros((len(y_pred_scaled), len(features) + 1))
        dummy_array[:, -1] = y_pred_scaled.flatten()
        y_pred_original_scale = scaler.inverse_transform(dummy_array)[:, -1]
        forecast_index = df.index[-horizon:]
        result.diff_log_pred[var] = pd.Series(y_pred_original_scale, index=forecast_index)
        log_series = df[f'log_{var}']
        last_log = log_series.iloc[-horizon - 1]
        log_pred = np.cumsum(result.diff_log_pred[var]) + last_log
        result.pred[var] = np.exp(log_pred) - 1e-6

        series = df[var]
        diff_log_series = df[f'diff_log_{var}']
        result.set_series_segments(var, series, diff_log_series, horizon)

        # Save fitted model and parameters
        result.fitted[var] = model
        # ...no parameter saving for RNN models...
        result.save_params_csv(var)
        # Call Diebold-Mariano test for each variable (GRU)
        result.diebold_mariano_test(var)

    return result


def create_sequences(data: np.ndarray, lag: int, input_cols_idx: list[int], target_col_idx: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)
