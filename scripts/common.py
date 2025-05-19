import json, os, random, sys
import numpy as np
from numpy import array
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf

class ResponseVariable:
    def __init__(self, 
                 title : str = None, 
                 forecast_index : DatetimeIndex = None,
                 past : Series = None, 
                 true : Series = None, 
                 pred : Series = None,
                 diff_log_past : Series = None, 
                 diff_log_true : Series = None, 
                 diff_log_pred : Series = None,
                 ):
        self.title = title
        self.forecast_index = forecast_index
        self.past = past
        self.true = true
        self.pred = pred
        self.diff_log_past = diff_log_past
        self.diff_log_true = diff_log_true
        self.diff_log_pred = diff_log_pred
        self.benchmarks : Benchmarks = None

    def __repr__(self):
        return (f"ResponseVariable(forecast_index={self.forecast_index}, "
                f"past={self.past}, true={self.true}, pred={self.pred}, "
                f"diff_log_past={self.diff_log_past}, diff_log_true={self.diff_log_true}, "
                f"diff_log_pred={self.diff_log_pred})")
    def bench(self): 
        try:
            y_true : array = self.true.values
            y_pred : array = self.pred.values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mask = y_true != 0
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        except: 
            rmse, mae, r2, mape, smape = None, None, None, None, None
        
        if self.diff_log_true.empty == False and self.diff_log_pred.empty == False:
            diff_log_y_true : array = self.diff_log_true.values
            diff_log_y_pred : array = self.diff_log_pred.values
            diff_log_rmse = np.sqrt(mean_squared_error(diff_log_y_true, diff_log_y_pred))
            diff_log_mae = mean_absolute_error(diff_log_y_true, diff_log_y_pred)
            diff_log_r2 = r2_score(diff_log_y_true, diff_log_y_pred)
            diff_log_mask = diff_log_y_true != 0
            diff_log_denominator = (np.abs(diff_log_y_true) + np.abs(diff_log_y_pred)) / 2
            diff_log_mape = np.mean(np.abs((diff_log_y_true[diff_log_mask] - diff_log_y_pred[diff_log_mask]) / diff_log_y_true[diff_log_mask])) * 100
            diff_log_smape = np.mean(np.abs(diff_log_y_true[diff_log_mask] - diff_log_y_pred[diff_log_mask]) / diff_log_denominator[diff_log_mask]) * 100
            self.benchmarks = Benchmarks(rmse, mae, r2, mape, smape, diff_log_rmse, diff_log_mae, diff_log_r2, diff_log_mape, diff_log_smape)
        
        else:
            self.benchmarks = Benchmarks(rmse, mae, r2, mape, smape)

    def plot(self, show_past: bool = False, diff: bool = False):
        """
        Plots the forecast, optionally including historical data.

        Args:
            show_past (bool, optional): If True, includes historical data in the plot. Defaults to False.
            diff (bool, optional): If True, plots the differenced log values. Defaults to False.
            title_suffix (str, optional): Additional text for the plot title. Defaults to "".
        """
        if self.forecast_index is None or (self.true is None and self.pred is None):
            print("Error: forecast_index and at least one of 'true' or 'pred' must be set.")
            return

        actual = self.diff_log_true if diff and self.diff_log_true is not None else self.true
        predicted = self.diff_log_pred if diff and self.diff_log_pred is not None else self.pred
        history = self.diff_log_past if diff and self.diff_log_past is not None and show_past else self.past if show_past else None

        plt.figure(figsize=(12, 6) if show_past else (10, 5))

        if show_past:
            if self.past is None:
                print("Warning: 'past' data is not available, cannot show historical data.")
            else:
                n = 40  # Number of history observations to be displayed
                try:
                    history_to_plot = history.iloc[-n:]
                except:
                    history_to_plot = pd.Series(history[-n:])
                plt.plot(history_to_plot.index, history_to_plot, label=f'Historical', alpha=0.7)

        if actual is not None:
            plt.plot(self.forecast_index, actual, label=f'Actual', marker='o')
        if predicted is not None:
            plt.plot(self.forecast_index, predicted, label=f'Predicted', marker='x')

        title = f"{self.title}"
        if show_past:
            title += f" (showing {n} history)"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel('')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename_suffix = "history" if show_past else ""
        # plt.savefig(f"results/forecast/png/{title_suffix}, {var0}, {var1} {filename_suffix}.png")
        plt.show()

class Benchmarks:
    def __init__(self, 
                 rmse : float,
                 mae : float, 
                 r2 : float,
                 mape : float,
                 smape : float,
                 diff_log_rmse : float = None,
                 diff_log_mae : float = None, 
                 diff_log_r2 : float = None,
                 diff_log_mape : float = None,
                 diff_log_smape : float = None,
                 ):
        self.rmse = rmse
        self.mae = mae
        self.r2 = r2
        self.mape = mape
        self.smape = smape
        self.diff_log_rmse = diff_log_rmse
        self.diff_log_mae = diff_log_mae
        self.diff_log_r2 = diff_log_r2
        self.diff_log_mape = diff_log_mape
        self.diff_log_smape = diff_log_smape


    def __repr__(self):
        text : str = ''
        if self.diff_log_rmse:
            text += f"\nDiff Log RMSE:  {self.diff_log_rmse:.4f}"
            text += f"\nDiff Log MAE:   {self.diff_log_mae:.4f}"
            text += f"\nDiff Log R²:    {self.diff_log_r2:.4f}"
            text += f"\nDiff Log MAPE:  {self.diff_log_mape:.4f}%"
            text += f"\nDiff Log sMAPE: {self.diff_log_smape:.4f}%"
        if self.rmse:
            text += f"\nForecast RMSE:  {self.rmse:.4f}"
            text += f"\nForecast MAE:   {self.mae:.4f}"
            text += f"\nForecast R²:    {self.r2:.4f}"
            text += f"\nForecast MAPE:  {self.mape:.4f}%"
            text += f"\nForecast sMAPE: {self.smape:.4f}%"
        return text
    
    def to_dict(self):
        dic = {}
        if self.diff_log_rmse:
            dic['diff_log_rmse'] = self.diff_log_rmse
            dic['diff_log_mae'] = self.diff_log_mae
            dic['diff_log_r2'] = self.diff_log_r2
            dic['diff_log_mape'] = self.diff_log_mape
            dic['diff_log_smape'] = self.diff_log_smape
        dic['rmse'] = self.rmse
        dic['mae'] = self.mae
        dic['r2'] = self.r2
        dic['mape'] = self.mape
        dic['smape'] = self.smape
        return dic


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

def diff_df(df:DataFrame, var0 = None, var1 = None):
    """
    Read JSON file as a DataFrame
    Args:
        df (Dataframe): Dataframe
        var0 (String): Variable 0
        var1 (String): Variable 1
    """

    # Add diffrential data
    if var0:
        df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
        df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
    if var1:
        df[f'log_{var1}'] = np.log(df[var1] + 1e-6)
        df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()
    df.dropna(inplace=True)
    
    return df

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
    plt.savefig(f"results/forecast/png/{title_suffix}, {var0}, {var1}.png")  
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
        history = pd.Series(history[-n:])  

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

    plt.savefig(f"results/forecast/png/{title_suffix}, {var0}, {var1} history.png")  
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