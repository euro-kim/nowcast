import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
import os
import tensorflow as tf
import pandas as pd
import sys
import json
def init_df(data_file):
    """
    Read JSON file as a DataFrame
    Args:
        data_file (String): path to JSON file
    """
    try:
        df = pd.read_json(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in data file {data_file}")
        sys.exit(1)

    # Convert 'time' to datetime format
    df['time'] = df['time'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m')

    # Set the 'time' column as the index
    df.set_index('time', inplace=True)

    # Set frequency to monthly start (note: this assumes the data is monthly)
    df = df.asfreq('MS')
    
    return df

def evaluate_forecast(y_true, y_pred, var_name):
    """
    Evaluates and prints the forecast performance metrics.

    Args:
        y_true (pd.Series or np.ndarray): Actual values.
        y_pred (pd.Series or np.ndarray): Predicted values.
        var_name (str): Name of the variable being forecasted (for printing).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{var_name} Forecast RMSE: {rmse:.4f}")
    print(f"{var_name} Forecast MAE: {mae:.4f}")
    print(f"{var_name} Forecast R²: {r2:.4f}")
    
    return rmse, mae, r2

def plot_forecast(diff:bool, actual, predicted, var0, var1, forecast_index, title_suffix=""):
    """
    Plots the actual and predicted values.

    Args:
        actual (pd.Series): Actual values.
        predicted (pd.Series): Predicted values.
        var_name (str): Name of the variable being forecasted.
        forecast_index (pd.DatetimeIndex): Index for the forecast period.
        title_suffix (str, optional): Additional text for the plot title.
    """
    if diff:
        var0=f"diff_log_{var0}"
        var1=f"diff_log_{var1}"

    plt.figure(figsize=(10, 5))
    plt.plot(forecast_index, actual, label=f'Actual {var0}', marker='o')
    plt.plot(forecast_index, predicted, label=f'{title_suffix} Predicted {var0}', marker='x')
    plt.title(f"{var0} Forecast with {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(var0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/forecast/{title_suffix}, {var0}, {var1}.png")  
    plt.show()

def plot_forecast_past(diff:bool, history, actual, predicted, var0, var1, forecast_index, title_suffix=""):
    """
    Plots the historical data, actual values, and predicted values.

    Args:
        history (pd.Series): Historical values used for training.
        actual (pd.Series): Actual values for the forecast period.
        predicted (pd.Series): Predicted values.
        var_name (str): Name of the variable being forecasted.
        forecast_index (pd.DatetimeIndex): Index for the forecast period.
        title_suffix (str, optional): Additional text for the plot title.
    """
    if diff:
        var0=f"diff_log_{var0}"
        var1=f"diff_log_{var1}"

    n = 40 # number of history observations to be displayed
    try:
        history = history.iloc[-n:]
    except:
        history = history[-n:]

    plt.figure(figsize=(12, 6))
    plt.plot(history.index, history, label=f'Historical {var0}', alpha=0.7)
    plt.plot(forecast_index, actual, label=f'Actual {var0}', marker='o')
    plt.plot(forecast_index, predicted, label=f'{title_suffix} Predicted {var0}', marker='x')
    plt.title(f"{var0} Forecast with {title_suffix} (showing {n} history)")
    plt.xlabel("Date")
    plt.ylabel(var0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"results/forecast/{title_suffix}, {var0}, {var1} history.png")  
    plt.show()


def set_seeds(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def create_sequences(data, lag, input_cols_idx, target_col_idx):
    """Creates sequences for time series prediction."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)