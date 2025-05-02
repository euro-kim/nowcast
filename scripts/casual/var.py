import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import io
import sys


# Normalize the data to [0, 1]
def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else np.zeros_like(x)

# Define ADF test function
def adf_test(series, name):
    result = adfuller(series.dropna())
    return f'{name} ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}'

# Read your JSON file as a DataFrame
df = pd.read_json("assets/data.json")

# Convert 'time' to datetime format
df['time'] = df['time'].astype(str)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m')

# Set the 'time' column as the index
df.set_index('time', inplace=True)

# # Check and handle duplicates
# duplicates = df.index[df.index.duplicated()]
# print(f"Duplicate dates: {duplicates}")

# Set frequency to monthly start ('MS')
df = df.asfreq('MS')

original_features = ['cpi', 'ppi']
var0 = original_features[0]
var1 = original_features[1]

# Normalization
series_var0 = min_max_norm(df[var0].values.astype(float))
series_var1 = min_max_norm(df[var1].values.astype(float))

# Correlation and Distance
corr, _ = pearsonr(series_var0, series_var1)
dtw_distance = dtw.distance(series_var0, series_var1)

# Log transformation (avoid log0)
df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
df[f'log_{var1}'] = np.log(df[var1] + 1e-6)  

# Differencing to make data stationary
df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()

# Define Parameters
diff_log_features=[f'diff_log_{var0}', f'diff_log_{var1}']

maxlags=15

# Prepare data for VAR model (drop any missing values)
path='results/casual/'
filename = 'var, '
filename+=', '.join(diff_log_features)
model_data = df[diff_log_features].dropna()


# Fit VAR model
model = VAR(model_data)
regression_results = model.fit(maxlags=maxlags, ic='aic')
lageval_results = model.select_order(maxlags=maxlags)  
causality_results = grangercausalitytests(model_data[diff_log_features], maxlag=maxlags, verbose=True)


with open(f"{path}{filename}.txt", "w") as f:
    f.write('--- Pearson Correlation ---\n')
    f.write(f"Normalized Pearson Correlation: {corr}\n")
    f.write(f"Normalized DTW Distance: {dtw_distance}\n")

    f.write('\n')
    f.write('\n')
    f.write('\n--- ADF Statistics ---\n')
    for feature in diff_log_features:
        p_value = adf_test(df[feature], feature)
        f.write(f"{p_value}\n")

    f.write('\n')
    f.write('\n')
    f.write('\n--- Regression Summary ---\n')
    f.write(str(regression_results.summary()))

    f.write('\n--- Lag Order Selection ---\n')
    f.write('Lag\tAIC\t\t\tBIC\t\t\tHQIC\n')

    # Extract values
    aic_vals = lageval_results.ics['aic']
    bic_vals = lageval_results.ics['bic']
    hqic_vals = lageval_results.ics['hqic']

    # Find the best (min) index
    best_aic = lageval_results.aic
    best_bic = lageval_results.bic
    best_hqic = lageval_results.hqic

    for lag in range(1, len(aic_vals) + 1):
        aic = f"{aic_vals[lag-1]:.6f}" + (" *" if lag == best_aic else "")
        bic = f"{bic_vals[lag-1]:.6f}" + (" *" if lag == best_bic else "")
        hqic = f"{hqic_vals[lag-1]:.6f}" + (" *" if lag == best_hqic else "")
        f.write(f"{lag:<4}\t{aic:<16}\t{bic:<16}\t{hqic}\n")

    f.write('\n')
    f.write('\n')
    f.write('\n--- Granger Causality Tests ---\n')
    response=''
    explanatory=''
    for index, feature in enumerate(diff_log_features):
        if index==0: response = feature
        else: 
            if index>1: explanatory+=', '
            explanatory+=feature
    f.write(f"Testing if {explanatory} Granger-causes {response}:")
    # Capture printed output of grangercausalitytests
    old_stdout = sys.stdout
    sys.stdout = mystdout = io.StringIO()
    grangercausalitytests(model_data[diff_log_features], maxlag=maxlags, verbose=True)
    sys.stdout = old_stdout
    f.write(mystdout.getvalue())
