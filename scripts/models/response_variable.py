import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scripts.models.benchmarks import Benchmarks

class ResponseVariable:
    def __init__(self):
        self.model_type = ""
        self.model_name = ""  
        self.forecast_index = None
        self.variables = []
        self.past = {}
        self.true = {}
        self.diff_log_past = {}
        self.diff_log_true = {}
        self.diff_log_pred = {}
        self.pred = {}
        self.benchmarks = []
        self.fitted = {}   
        self.params = {}   

    def set_variables(self, variables):
        self.variables = variables
        for v in variables:
            self.past[v] = None
            self.true[v] = None
            self.diff_log_past[v] = None
            self.diff_log_true[v] = None
            self.diff_log_pred[v] = None
            self.pred[v] = None
        # Save params CSV for each variable if params already set (e.g., after loading)
        for v in variables:
            if v in self.params:
                self.save_params_csv(v)

    def save_params_csv(self, var):
        """
        Save the parameters for a given variable to a CSV file in the appropriate directory.
        Assumes self.params[var] is a list of dicts, each with keys:
        variable, model, order, parameter, value.
        For LSTM/GRU, saves the trained model using model.save().
        """
        import pandas as pd
        import os

        outdir = f"results/{'&'.join(self.variables)}/predict-{var}"
        os.makedirs(outdir, exist_ok=True)

        # Save Keras model for LSTM/GRU
        if self.model_type in ("LSTM", "GRU"):
            model = self.fitted.get(var)
            if model is not None:
                model_path = f"{outdir}/{self.model_name}.keras"
                model.save(model_path)
            return

        if var not in self.params or self.params[var] is None:
            print("Warning: No parameters to save for variable", var)
            return

        rows = self.params[var]
        if not isinstance(rows, list) or not rows:
            print("Warning: No parameters to save for variable", var)
            return
        print(f"Saving parameters for {var} to {outdir}/{self.model_name}.csv")
        fname = f"{outdir}/{self.model_name}.csv"
        df = pd.DataFrame(rows)
        df.to_csv(fname, index=False)

    def set_series_segments(self, var, series, diff_log_series, horizon):
        """
        Utility to set past, true, diff_log_past, and diff_log_true for a variable.
        """
        self.past[var] = series.iloc[:-(horizon)]
        self.true[var] = series.iloc[-horizon:]
        self.diff_log_past[var] = diff_log_series[:-horizon].dropna()
        self.diff_log_true[var] = diff_log_series[-horizon:].dropna()

    def __repr__(self):
        return (f"ResponseVariable(forecast_index={self.forecast_index}, "
                f"past={self.past}, true={self.true}, pred={self.pred}, "
                f"diff_log_past={self.diff_log_past}, diff_log_true={self.diff_log_true}, "
                f"diff_log_pred={self.diff_log_pred})")
    def bench(self): 
        import csv
        import pandas as pd
        variables = list(self.variables) if hasattr(self, 'variables') else []
        for var in variables:
            self.target_var = var
            try:
                y_true = self.true[var].values
                y_pred = self.pred[var].values
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                mask = y_true != 0
                denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            except Exception:
                rmse = mae = r2 = mape = smape = None

            # --- diff_log metrics ---
            diff_log_rmse = diff_log_mae = diff_log_r2 = diff_log_mape = diff_log_smape = None
            try:
                if self._valid_diff_log(var):
                    diff_log_y_true = self.diff_log_true[var].values
                    diff_log_y_pred = self.diff_log_pred[var].values
                    mask = ~np.isnan(diff_log_y_true) & ~np.isnan(diff_log_y_pred)
                    diff_log_y_true = diff_log_y_true[mask]
                    diff_log_y_pred = diff_log_y_pred[mask]
                    if len(diff_log_y_true) > 0:
                        diff_log_rmse = np.sqrt(mean_squared_error(diff_log_y_true, diff_log_y_pred))
                        diff_log_mae = mean_absolute_error(diff_log_y_true, diff_log_y_pred)
                        diff_log_r2 = r2_score(diff_log_y_true, diff_log_y_pred)
                        diff_log_mask = diff_log_y_true != 0
                        diff_log_denominator = (np.abs(diff_log_y_true) + np.abs(diff_log_y_pred)) / 2
                        diff_log_mape = np.mean(np.abs((diff_log_y_true[diff_log_mask] - diff_log_y_pred[diff_log_mask]) / diff_log_y_true[diff_log_mask])) * 100
                        diff_log_smape = np.mean(np.abs(diff_log_y_true[diff_log_mask] - diff_log_y_pred[diff_log_mask]) / diff_log_denominator[diff_log_mask]) * 100
            except Exception:
                pass

            bm = Benchmarks(
                rmse, mae, r2, mape, smape,
                diff_log_rmse, diff_log_mae, diff_log_r2, diff_log_mape, diff_log_smape,
                target_variable=var
            )
            self.benchmarks.append(bm)

            outdir = f"results/{'&'.join(variables)}/predict-{var}"
            os.makedirs(outdir, exist_ok=True)
            fname = f"{outdir}/benchmark.csv"
            model_name = self.model_name if hasattr(self, 'model_name') else 'model'
            bench_dict = bm.to_dict()
            if not os.path.exists(fname):
                with open(fname, "w", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["metric", model_name])
                    for k, v in bench_dict.items():
                        writer.writerow([k, v])
                        
            else:
                df = pd.read_csv(fname, index_col=0)
                if model_name in df.columns:
                    print(f"Model '{model_name}' already exists in {fname}, skipping update.")
                    continue
                new_col = pd.Series(bench_dict, name=model_name)
                df = df.join(new_col, how='outer')
                df.index.name = "metric"
                df.to_csv(fname)

    def _valid_diff_log(self, v):
        return (
            self.diff_log_true.get(v) is not None and self.diff_log_pred.get(v) is not None
            and not pd.isnull(self.diff_log_true[v]).all() and not pd.isnull(self.diff_log_pred[v]).all()
            and len(self.diff_log_true[v]) > 0 and len(self.diff_log_pred[v]) > 0
        )

    def _concat_valid_diff_logs(self, valid_vars):
        diff_log_y_true = np.concatenate([self.diff_log_true[v].values for v in valid_vars])
        diff_log_y_pred = np.concatenate([self.diff_log_pred[v].values for v in valid_vars])
        mask = ~np.isnan(diff_log_y_true) & ~np.isnan(diff_log_y_pred)
        return diff_log_y_true[mask], diff_log_y_pred[mask]

    def plot(self, show_past: bool = False, diff: bool = False):
        """
        Plots the forecast, optionally including historical data.

        Args:
            show_past (bool, optional): If True, includes historical data in the plot. Defaults to False.
            diff (bool, optional): If True, plots the differenced log values. Defaults to False.
            notes (str, optional): Additional notes to display on the plot. Defaults to "".
        """
        if self.forecast_index is None or (all(v is None for v in self.true.values()) and all(v is None for v in self.pred.values())):
            print("Error: forecast_index and at least one of 'true' or 'pred' must be set.")
            return

        for i, var in enumerate(self.variables):
            actual = self.diff_log_true[var] if diff and self.diff_log_true[var] is not None else self.true[var]
            predicted = self.diff_log_pred[var] if diff and self.diff_log_pred[var] is not None else self.pred[var]
            history = self.diff_log_past[var] if diff and self.diff_log_past[var] is not None and show_past else self.past[var] if show_past else None

            plt.figure(figsize=(12, 6) if show_past else (10, 5))

            if show_past:
                if self.past[var] is None:
                    print(f"Warning: 'past' data is not available for {var}, cannot show historical data.")
                else:
                    n = 40  # Number of history observations to be displayed
                    try:
                        history_to_plot = history.iloc[-n:]
                    except:
                        history_to_plot = pd.Series(history[-n:])
                    plt.plot(history_to_plot.index, history_to_plot, label=f'Historical', alpha=0.7, color='black', linewidth=2, linestyle='--')

            if diff:
                diff_suffix = 'diff_log_'
            else:
                diff_suffix = 'restored_'
            if show_past:
                suffix = diff_suffix + 'history'
            else:
                suffix = diff_suffix + 'horizon'

            if actual is not None:
                plt.plot(self.forecast_index, actual, label=f'Actual', marker='o', color='black', linewidth=2, alpha=0.7)
            if predicted is not None:
                plt.plot(self.forecast_index, predicted, label=f'Predicted', marker='x', color='red', linewidth=2)

            outdir = f"results/{'&'.join(self.variables)}/predict-{var}"
            os.makedirs(outdir, exist_ok=True)
            fname = f"{outdir}/{self.model_name}_{suffix}.png"

            title = f"{self.model_name} Forecast - Predicting {'log_diff_' if diff else ''}{var} {'(Restored Scale)' if not diff else ''}"
            # if show_past:
            #     notes = f" Showing {n} history and forecast horizon\n"
            # else:
            #     notes = f" Showing forecast horizon only\n"

            variables = [f"{circled_number(index+1)} diff_log_{str(variable)}" for index, variable in enumerate(self.variables)]
            notes = f"{len(variables)} variables used:\n{'\n'.join(variables)}"
            plt.title(title, fontsize=18, fontweight='bold')
            plt.xlabel("Date", fontsize=14)
            plt.ylabel(f'{diff_suffix.replace("restored_","")}{var}', fontsize=14)
            plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
            plt.grid(True, linestyle='--', color='gray', alpha=0.4)
            plt.xticks(rotation=30, fontsize=11)
            plt.yticks(fontsize=11)
            # Place annotation inside the plot area, upper left
            plt.annotate(
                notes,
                xy=(0.01, 0.99), xycoords='axes fraction',
                fontsize=12, color="black",
                xytext=(0, 0), textcoords='offset points',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
            )
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight')
            # plt.show()
    def diebold_mariano_test(self, var, loss='mse'):
        """
        Perform Diebold-Mariano test for VAR, LSTM, or GRU models.
        Uses naive forecast as benchmark.
        Saves the DM statistic and p-value to a CSV in the results directory.
        If the file exists, appends a new row.
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        # Only run for VAR, LSTM, GRU
        if not (self.model_name.startswith("VAR") or self.model_name.startswith("LSTM") or self.model_name.startswith("GRU")):
            print(f"Diebold-Mariano test not applicable for model {self.model_name}")
            return

        # Get forecast errors
        y_true = self.true[var].values
        y_pred = self.pred[var].values

        # Naive forecast: previous value as prediction
        if hasattr(self, "past") and self.past.get(var) is not None:
            past = self.past[var].values
            naive_pred = np.concatenate([[past[-1]], y_true[:-1]])
        else:
            naive_pred = np.roll(y_true, 1)
            naive_pred[0] = y_true[0]

        if len(y_true) != len(y_pred) or len(y_true) != len(naive_pred):
            print("Error: Length mismatch for DM test.")
            return

        # Loss differential
        if loss == 'mse':
            d = (y_true - y_pred) ** 2 - (y_true - naive_pred) ** 2
        elif loss == 'mae':
            d = np.abs(y_true - y_pred) - np.abs(y_true - naive_pred)
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

        mean_d = np.mean(d)
        n = len(d)
        var_d = np.var(d, ddof=1)
        dm_stat = mean_d / np.sqrt(var_d / n) if var_d > 0 else np.nan

        # Two-sided p-value
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))

        # Save or append results
        outdir = f"results/{'&'.join(self.variables)}/predict-{var}"
        os.makedirs(outdir, exist_ok=True)
        fname = f"{outdir}/dmtest.csv"
        new_row = {
            "model": self.model_name,
            "variable": var,
            "loss": loss,
            "dm_stat": dm_stat,
            "p_value": p_value
        }
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(fname, index=False)
        else:
            df = pd.DataFrame([new_row])
            df.to_csv(fname, index=False)
        print(f"Diebold-Mariano test saved to {fname}")
# Circled numbers for 1-20: ①-⑳ (Unicode 9312-9331)

def circled_number(n):
    if 1 <= n <= 20:
        return chr(9311 + n)  # 9312 is ①
    return str(n)