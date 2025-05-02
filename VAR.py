
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Read your JSON file as a DataFrame
df = pd.read_json("data.json")

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
# Assume you have prepared your data as mentioned earlier.
df[f'log_{var0}'] = np.log(df[var0] + 1e-6)
df[f'log_{var1}'] = np.log(df[var1] + 1e-6)


# Differencing to make data stationary
df[f'diff_log_{var0}'] = df[f'log_{var0}'].diff()
df[f'diff_log_{var1}'] = df[f'log_{var1}'].diff()

# Define parameters
forecast_steps = 12
diff_log_features = [f'diff_log_{var0}', f'diff_log_{var1}']


# Prepare data for the VAR model (drop NaN values after differencing)
model_data = df[diff_log_features].dropna()

# Split the data into training and test sets
train_data = model_data[:-forecast_steps]
test_data = model_data[-forecast_steps:]

# Fit the VAR model
model = VAR(train_data)
maxlags = 15  # Adjust this based on your optimal lag selection
regression_results = model.fit(maxlags=maxlags, ic='aic')

# Forecasting using the fitted model
lag_order = regression_results.k_ar  # Get the optimal lag order
forecast_input = train_data.values[-lag_order:]  # Take the last `lag_order` observations
forecast_diff_log = regression_results.forecast(y=forecast_input, steps=forecast_steps)

# Create a DataFrame for the forecasted differenced log values
forecast_df = pd.DataFrame(forecast_diff_log, columns=diff_log_features)

# Reconstruct the log of var0 (since you are predicting var0)
last_log_var0 = df[f'log_{var0}'].iloc[-forecast_steps - 1]  # Last value of log_var0 before forecast

# Convert differenced log values back to log values
forecast_df[f'log_{var0}'] = forecast_df[f'diff_log_{var0}'].cumsum() + last_log_var0

# Convert log_var0 to original var0 scale
forecast_df[f'{var0}_forecast'] = np.exp(forecast_df[f'log_{var0}']) - 1e-6  # Apply inverse of log transformation

# Get actual values of var0 for comparison (from the test set)
actual_var0 = df[var0].iloc[-forecast_steps:]

# Align forecast with actuals for evaluation
forecast_df.index = actual_var0.index
y_true = actual_var0.values
y_pred = forecast_df[f'{var0}_forecast'].values

# Evaluate the forecast performance
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Forecast RMSE: {rmse:.4f}")
print(f"Forecast MAE: {mae:.4f}")
print(f"Forecast R²: {r2:.4f}")

import matplotlib.pyplot as plt
# Plot
forecast_index = df.index[-forecast_steps:]
plt.figure(figsize=(10, 5))
plt.plot(forecast_index, y_true, label='Actual CPI', marker='o')
plt.plot(forecast_index, y_pred, label='VAR Predicted CPI', marker='x')
plt.title(f"{var0.upper()} Forecast using {var1.upper()} (GRU)")
plt.xlabel("Date")
plt.ylabel(var0.upper())
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
