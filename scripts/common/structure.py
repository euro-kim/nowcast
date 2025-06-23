import json, sys
import numpy as np
import pandas as pd

def min_max_norm(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else np.zeros_like(x)
    
def init_df(data_file):
    """
    Read JSON or CSV file as a DataFrame
    Args:
        data_file (String): path to JSON or CSV file
    """
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            df = pd.read_json(data_file)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError):
        print(f"Error: Invalid file format in data file {data_file}")
        sys.exit(1)

    # Convert 'time' to datetime format
    df['time'] = df['time'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m')

    # Set the 'time' column as the index
    df.set_index('time', inplace=True)

    # Set frequency to monthly start (note: this assumes the data is monthly)
    df = df.asfreq('MS')

    return df

def diff_df(df, *vars):
    """
    Add log and diff_log columns for each variable in vars.
    """
    eps = 1e-6  # Small constant to avoid log(0)
    for var in vars:
        df[f'log_{var}'] = np.log(df[var] + eps)
        df[f'diff_log_{var}'] = df[f'log_{var}'].diff()
    return df

def create_sequences(data, lag, input_cols_idx, target_col_idx):
    """Creates sequences for time series prediction."""
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)
