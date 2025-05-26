import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score
import properscoring as ps
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import sys

class Benchmarks:
    def __init__(self, mae: float = None, rmse: float = None, mape: float = None, smape: float = None, r2: float = None, crps: float = None):
        self.mae = mae
        self.rmse = rmse
        self.mape = mape
        self.smape = smape
        self.r2 = r2
        self.crps = crps

    def __repr__(self):
        metrics_str = []
        if self.mae is not None:
            metrics_str.append(f"MAE: {self.mae:.3f}")
        if self.rmse is not None:
            metrics_str.append(f"RMSE: {self.rmse:.3f}")
        if self.mape is not None:
            metrics_str.append(f"MAPE: {self.mape:.2f}%")
        if self.smape is not None:
            metrics_str.append(f"sMAPE: {self.smape:.2f}%")
        if self.r2 is not None:
            metrics_str.append(f"R2: {self.r2:.3f}")
        if self.crps is not None:
            metrics_str.append(f"CRPS: {self.crps:.3f}")
        return ", ".join(metrics_str)

class ResponseVariable:
    def __init__(self,
                 title: str = None,
                 past: Series = None,
                 true: Series = None,
                 pred: Series = None,
                 forecast_index: pd.Index = None 
                 ):
        self.title = title
        self.past = past
        self.true = true
        self.pred = pred

        self.forecast_index = forecast_index

        self.benchmarks = Benchmarks() 

    def __repr__(self):
        data_summary = []
        if self.title: data_summary.append(f"Title: {self.title}")
        if self.past is not None: data_summary.append(f"Past (len={len(self.past)})")
        if self.true is not None: data_summary.append(f"True (len={len(self.true)})")
        if self.pred is not None: data_summary.append(f"Pred (len={len(self.pred)})")
        if self.forecast_index is not None: data_summary.append(f"Forecast Index (len={len(self.forecast_index)})")

        return f"<ResponseVariable: {', '.join(data_summary)}>\nBenchmarks: {self.benchmarks}"

    def calculate_benchmarks(self, y_true_orig_scale: np.ndarray, y_pred_orig_scale: np.ndarray, y_true_log_scale: np.ndarray = None, y_pred_log_scale: np.ndarray = None, lower_bound: np.ndarray = None, upper_bound: np.ndarray = None):
        """
        Calculates and stores various performance metrics in the benchmarks attribute.
        y_true_orig_scale, y_pred_orig_scale are required for MAE, RMSE.
        y_true_log_scale, y_pred_log_scale are for MAPE, sMAPE (if based on original log scale).
        lower_bound, upper_bound for CRPS calculation.
        """
        self.benchmarks.mae = mean_absolute_error(y_true_orig_scale, y_pred_orig_scale)
        self.benchmarks.rmse = np.sqrt(mean_squared_error(y_true_orig_scale, y_pred_orig_scale))

        # Calculate MAPE and sMAPE if log scale values are provided
        if y_true_log_scale is not None and y_pred_log_scale is not None:
            self.benchmarks.mape = self._MAPE(y_true_log_scale, y_pred_log_scale)
            self.benchmarks.smape = self._sMAPE(y_true_log_scale, y_pred_log_scale)

        # Calculate R2 Score
        self.benchmarks.r2 = r2_score(y_true_orig_scale, y_pred_orig_scale)

        # Calculate CRPS if prediction intervals are available
        if lower_bound is not None and upper_bound is not None:
            # Assuming Gaussian distribution for CRPS with median and std dev derived from bounds
            # std dev for 90% CI (5%-95%) is approx (upper - lower) / (2 * 1.645) = (upper - lower) / 3.29
            # For 95% CI (2.5%-97.5%), it's approx (upper - lower) / (2 * 1.96) = (upper - lower) / 3.92
            # Here, the code uses / 4, which is roughly for ~95% CI if assuming normal distribution
            # You might want to adjust this based on the actual quantile levels [0.05, 0.95] for 90% CI.
            # For a 90% CI, the standard deviation is (upper - lower) / (2 * 1.645) ~ (upper - lower) / 3.29
            # If your quantiles are [0.05, 0.95], this is a 90% interval, so 3.29 is more appropriate.
            # The original code uses / 4, which is closer to a 95% interval if it were 2.5% and 97.5%.
            # Let's stick with / 4 for now, assuming its intended meaning.
            crps_scores = [
                ps.crps_gaussian(y_true_orig_scale[i], mu=y_pred_orig_scale[i],
                                 sig=(upper_bound[i] - lower_bound[i]) / 4) # Adjust divisor based on quantile levels if needed
                for i in range(len(y_true_orig_scale))
            ]
            self.benchmarks.crps = np.mean(crps_scores)

    def _MAPE(self, y_true_log, y_pred_log):
        # Convert back to original scale for MAPE calculation
        y_true_orig = np.exp(y_true_log)
        y_pred_orig = np.exp(y_pred_log)
        mask = y_true_orig != 0
        return np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100

    def _sMAPE(self, y_true_log, y_pred_log):
        # Convert back to original scale for sMAPE calculation
        y_true_orig = np.exp(y_true_log)
        y_pred_orig = np.exp(y_pred_log)
        denominator = (np.abs(y_true_orig) + np.abs(y_pred_orig)) / 2
        mask = denominator != 0 # Avoid division by zero
        return np.mean(np.abs(y_true_orig[mask] - y_pred_orig[mask]) / denominator[mask]) * 100


def MAPE_orig_scale(y_true_orig, y_pred_orig):
    # MAPE directly on original scale values 
    mask = y_true_orig != 0
    return np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100
