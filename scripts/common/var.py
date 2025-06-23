import numpy as np
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import io, sys
import os
import csv
from statsmodels.tsa.vector_ar.var_model import VARResults


def regression_summary(fitted_var_model: VARResults, variables: list[str]) -> str:
    """
    Save regression summary from a fitted VAR model to a file.
    variables: list of strings (variable names)
    fitted_var_model: a fitted VAR model (from statsmodels)
    """
    results = str(fitted_var_model.summary())
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/var_summary.txt"
    with open(fname, "w") as f:
        f.write(results)
    return fname

def lag_order_selection(fitted_var_model: VARResults, maxlags: int, variables: list[str]) -> str:
    """
    Save lag order selection table from a fitted VAR model to a CSV file.
    variables: list of strings (variable names)
    fitted_var_model: a fitted VAR model (from statsmodels)
    """
    lageval_results = fitted_var_model.select_order(maxlags=maxlags)
    aic_vals = lageval_results.ics['aic']
    bic_vals = lageval_results.ics['bic']
    hqic_vals = lageval_results.ics['hqic']
    best_aic = lageval_results.aic
    best_bic = lageval_results.bic
    best_hqic = lageval_results.hqic
    rows = []
    for lag in range(1, len(aic_vals) + 1):
        rows.append({
            'lag': lag,
            'aic': aic_vals[lag-1],
            'bic': bic_vals[lag-1],
            'hqic': hqic_vals[lag-1],
            'best_aic': lag == best_aic,
            'best_bic': lag == best_bic,
            'best_hqic': lag == best_hqic
        })
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/lag_order.csv"
    with open(fname, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lag', 'aic', 'bic', 'hqic', 'best_aic', 'best_bic', 'best_hqic'])
        writer.writeheader()
        writer.writerows(rows)
    return fname

def full_granger_causality(fitted_var_model: VARResults, variables: list[str], maxlags: int) -> str:
    """
    Run full Granger causality tests for all pairs of variables (log-diff) and save to file.
    variables: list of strings (variable names)
    fitted_var_model: a fitted VAR model (from statsmodels)
    """
    diff_log_features = fitted_var_model.names
    results = []
    if len(diff_log_features) == 2:
        # Use bivariate grangercausalitytests for two variables
        import pandas as pd
        from statsmodels.tsa.stattools import grangercausalitytests
        # Assume the original DataFrame is accessible as fitted_var_model.endog (numpy array)
        df = pd.DataFrame(fitted_var_model.endog, columns=diff_log_features, index=fitted_var_model.model.data.row_labels)
        for i, response in enumerate(diff_log_features):
            explanatory = [f for j, f in enumerate(diff_log_features) if j != i]
            results.append(f"\nTesting if {', '.join(explanatory)} Granger-causes {response}:")
            try:
                test_data = df[[response] + explanatory].dropna()
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                grangercausalitytests(test_data, maxlag=maxlags, verbose=True)
                sys.stdout = old_stdout
                results.append(mystdout.getvalue())
            except Exception as e:
                sys.stdout = old_stdout
                results.append(f"Granger causality test failed: {e}")
    else:
        # Use VAR model's test_causality for multivariate case
        for i, response in enumerate(diff_log_features):
            explanatory = [f for j, f in enumerate(diff_log_features) if j != i]
            results.append(f"\nTesting if {', '.join(explanatory)} Granger-causes {response}:")
            try:
                causality = fitted_var_model.test_causality(response, explanatory, kind='f')
                results.append(str(causality.summary()))
            except Exception as e:
                results.append(f"Granger causality test failed: {e}")
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/granger.txt"
    with open(fname, "w") as f:
        for line in results:
            f.write(line + "\n")
    return fname


