import numpy as np
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import io, sys
import os
import csv
def descriptive_stats(df: DataFrame, variables: list[str]) -> str:
    """
    Compute descriptive statistics for the given variables and save to CSV file.
    Statistics: Sample Size, MAX, MIN, Variance, Average, Q1, Q2, Q3, Std
    Each variable is a column, each statistic is a row.
    """
    stats = {
        "Sample Size": [],
        "MIN": [],
        "Q1": [],
        "Q2": [],
        "Q3": [],
        "MAX": [],
        "Average": [],
        "Variance": [],
        "Std": []
    }
    for var in variables:
        series = df[var].dropna()
        stats["Sample Size"].append(len(series))
        stats["MAX"].append(series.max())
        stats["Q1"].append(series.quantile(0.25))
        stats["Q2"].append(series.quantile(0.5))
        stats["Q3"].append(series.quantile(0.75))
        stats["MIN"].append(series.min())
        stats["Average"].append(series.mean())
        stats["Variance"].append(series.var())
        stats["Std"].append(series.std())

    stats_df = DataFrame(stats, index=variables).T
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/descriptive_stats.csv"
    stats_df.to_csv(fname)
    return fname

def adf_test(series: DataFrame, name: str) -> str:
    result = adfuller(series.dropna())
    return f'{name} ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}'

def adf_stats(df: DataFrame, variables: list[str]) -> str:
    """
    Save ADF statistics for each variable and their log-diff to a CSV file.
    variables: list of strings (variable names)
    """
    results = []
    for var in variables:
        stat, pval = adfuller(df[var].dropna())[0:2]
        results.append({'variable': var, 'adf_statistic': stat, 'p_value': pval})
    for var in variables:
        safe_vals = df[var].copy()
        safe_vals[safe_vals <= 0] = np.nan
        log_var = np.log(safe_vals + 1e-6)
        diff_log_var = log_var.diff()
        stat, pval = adfuller(diff_log_var.dropna())[0:2]
        results.append({'variable': f'diff_log_{var}', 'adf_statistic': stat, 'p_value': pval})
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/adf_stats.csv"

    with open(fname, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['variable', 'adf_statistic', 'p_value'])
        writer.writeheader()
        writer.writerows(results)
    return fname

def pearson_correlations(df: DataFrame, variables: list[str]) -> str:
    """
    Compute pairwise Pearson correlations for the given variables and save to CSV file.
    variables: list of strings (variable names)
    """
    results = []
    for i, var_i in enumerate(variables):
        for j, var_j in enumerate(variables):
            if i < j:
                valid = df[[var_i, var_j]].dropna()
                if len(valid) > 1:
                    corr, _ = pearsonr(valid[var_i], valid[var_j])
                    results.append({'pair': f'({var_i}, {var_j})', 'pearson_correlation': corr})
                else:
                    results.append({'pair': f'({var_i}, {var_j})', 'pearson_correlation': float('nan')})
    outdir = f"results/{'&'.join(variables)}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{outdir}/pearson.csv"
    with open(fname, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pair', 'pearson_correlation'])
        writer.writeheader()
        writer.writerows(results)
    return fname

# Remove any import of modules that import this file to avoid circular dependencies.
# Do NOT import from .forecast or any module that imports from this file.
