import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from prophet import Prophet
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error
import os
import sys
from models import ResponseVariable, MAPE_orig_scale 

# Create a directory to save results if it doesn't exist
output_dir = "results/prob1" # Corrected output directory to match original intent
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Korean Font
path = r'C:\Windows\Fonts\batang.ttc'
fontprop = fm.FontProperties(fname=path, size=10)
try:
    plt.rc('font', family=fontprop.get_name())
except FileNotFoundError:
    print(f"Warning: Font file not found at {path}. Using default font.")
    plt.rc('font', family='Malgun Gothic' if 'Malgun Gothic' in fm.findSystemFonts(fontpaths=None, fontext='ttf') else 'sans-serif')

# 1. 데이터 수집 및 전처리
print("=== 1. 데이터 수집 및 전처리 ===")

# Define start and end dates
start = '2005-01-01'
end = '2017-12-31'

# 1-1) Hourly Energy Consumption Data Load and Resample to Daily
energy_file_path = "AEP_hourly.csv"
try:
    df_energy_hourly = pd.read_csv(energy_file_path, parse_dates=['Datetime'])
except FileNotFoundError:
    print(f"Error: Energy data file '{energy_file_path}' not found. Exiting.")
    sys.exit(1)

# Set Datetime as index
df_energy_hourly = df_energy_hourly.set_index('Datetime')

# Resample to daily frequency and sum the hourly consumption
df_energy_daily = df_energy_hourly.resample('D').sum()
df_energy_daily.rename(columns={'AEP_MW': 'daily_consumption_MW'}, inplace=True)
df_energy_daily.index.name = 'date' # Explicitly name the index 'date'

# Filter data to the desired period
df_energy_daily = df_energy_daily[(df_energy_daily.index >= start) & (df_energy_daily.index <= end)]

# Interpolate any missing daily values that might arise from gaps in hourly data
df_energy_daily['daily_consumption_MW'] = df_energy_daily['daily_consumption_MW'].interpolate(method='linear', limit_direction='both')

# Add log-transformed consumption
if (df_energy_daily['daily_consumption_MW'] <= 0).any():
    print("Warning: Non-positive values found in 'daily_consumption_MW'. Adding a small constant before logging.")
    df_energy_daily['daily_consumption_MW'] = df_energy_daily['daily_consumption_MW'].apply(lambda x: max(x, 1e-6))

df_energy_daily['log_daily_consumption_MW'] = np.log(df_energy_daily['daily_consumption_MW'])

full_idx = pd.date_range(start=start, end=end, freq='D')
df_energy_daily = df_energy_daily.reindex(full_idx)
df_energy_daily.index.name = 'date' # Re-assert index name just in case reindex clears it
df_energy_daily['daily_consumption_MW'] = df_energy_daily['daily_consumption_MW'].interpolate(method='linear', limit_direction='both')
df_energy_daily['log_daily_consumption_MW'] = np.log(df_energy_daily['daily_consumption_MW']) # Re-log after interpolation if new NaNs were created

# Display energy data with 'date' as index
print("일별 에너지 소비 데이터 일부:\n", df_energy_daily.head())
print(f"일별 에너지 소비 데이터 길이: {len(df_energy_daily)}")


# 1-2) 외생변수로 온도 데이터 로드 및 전처리
temp_file_path = "GlobalLandTemperaturesByCity.csv"
try:
    df_temp_raw = pd.read_csv(temp_file_path, encoding='latin1', on_bad_lines='skip')
except FileNotFoundError:
    print(f"Error: Temperature data file '{temp_file_path}' not found. Exiting.")
    sys.exit(1)

df_temp_raw['dt'] = pd.to_datetime(df_temp_raw['dt'])

# 1. Filter for New York directly AND select only 'dt' and 'AverageTemperature' immediately
df_temp_ny = df_temp_raw[df_temp_raw['City']=='New York'][['dt', 'AverageTemperature']].copy()

# 2. Set 'dt' as index for time series operations
df_temp_ny = df_temp_ny.set_index('dt')
df_temp_ny.index.name = 'date' # Rename index to 'date' for consistency with energy data

# 3. Interpolate missing values (before reindexing to preserve original gaps)
df_temp_ny['AverageTemperature'] = df_temp_ny['AverageTemperature'].interpolate(method='linear', limit_direction='both')

# 4. Filter to the desired date range
df_temp_ny = df_temp_ny[(df_temp_ny.index >= start) & (df_temp_ny.index <= end)]

# 5. Reindex to the full_idx to ensure all dates are present and fill potential new NaNs
df_temp_ny = df_temp_ny.reindex(full_idx)
df_temp_ny.index.name = 'date' # Re-assert index name just in case reindex clears it
df_temp_ny['AverageTemperature'] = df_temp_ny['AverageTemperature'].interpolate(method='linear', limit_direction='both')


# Display temperature data with 'date' as index
print("뉴욕 온도 데이터 일부:\n", df_temp_ny.head())
print(f"뉴욕 온도 데이터 길이: {len(df_temp_ny)}")

# 1-3) 병합 - Merging energy data with temperature data
df_energy_to_merge = df_energy_daily.reset_index()
df_temp_to_merge = df_temp_ny.reset_index()

df_all = pd.merge(df_energy_to_merge, df_temp_to_merge[['date', 'AverageTemperature']], on='date', how='left')
df_all.rename(columns={'AverageTemperature':'temp'}, inplace=True)
df_all['temp'] = df_all['temp'].interpolate(method='linear', limit_direction='both')

# Prophet 데이터 준비
df_prophet = df_all[['date', 'log_daily_consumption_MW', 'temp']].rename(columns={'date':'ds', 'log_daily_consumption_MW':'y'})

# --- DIAGNOSTIC PRINTS START ---
print(f"\n--- Diagnostic Checks ---")
print(f"Length of df_prophet before split: {len(df_prophet)}")
if df_prophet.empty:
    print("ERROR: df_prophet is empty. Cannot perform train/test split. Exiting.")
    sys.exit(1)

# 1-4) train/test split
split_idx = int(len(df_prophet)*0.85)
train = df_prophet.iloc[:split_idx].copy()
test = df_prophet.iloc[split_idx:].copy()
horizon = len(test)

print(f"Train 데이터 길이: {len(train)}, Test 데이터 길이: {len(test)}")

if test.empty:
    print("ERROR: Test DataFrame is empty. Cannot proceed with modeling or plotting. Exiting.")
    sys.exit(1)
# --- DIAGNOSTIC PRINTS END ---

# 2. Prophet 기본 모델 적합 및 평가
print("\n=== 2. Prophet 기본 모델 적합 및 평가 ===")
from pandas.tseries.holiday import USFederalHolidayCalendar

cal = USFederalHolidayCalendar()
holidays_dates = cal.holidays(start=start, end=end)
holidays = pd.DataFrame({
    'holiday': 'us_federal_holiday',
    'ds': holidays_dates,
    'lower_window': 0,
    'upper_window': 1,
})

prophet_base_results = ResponseVariable(title="Prophet Base Model")

model_base = Prophet(
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model_base.fit(train[['ds','y']])

# Check if test data is valid before predicting
if test[['ds']].empty:
    print("WARNING: Test data for Prophet base model is empty. Skipping prediction.")
    forecast_base = pd.DataFrame() # Create an empty DataFrame to avoid errors
else:
    forecast_base = model_base.predict(test[['ds']])

# Ensure forecast_base has 'yhat' and is not empty before proceeding
if 'yhat' not in forecast_base.columns or forecast_base.empty:
    print("ERROR: Prophet forecast_base does not contain 'yhat' column or is empty. Model prediction might have failed. Exiting.")
    sys.exit(1)


prophet_base_results.past = df_all['daily_consumption_MW'].iloc[:split_idx]
prophet_base_results.true = df_all['daily_consumption_MW'].iloc[split_idx:]
prophet_base_results.pred = pd.Series(np.exp(forecast_base['yhat'].values), index=test['ds'])
prophet_base_results.forecast_index = test['ds']

# Calculate benchmarks
prophet_base_results.calculate_benchmarks(
    y_true_orig_scale=prophet_base_results.true.values,
    y_pred_orig_scale=prophet_base_results.pred.values,
    y_true_log_scale=test['y'].values,
    y_pred_log_scale=forecast_base['yhat'].values
)

print(f"Prophet 기본 모델 - {prophet_base_results.benchmarks}")
with open(os.path.join(output_dir, "prophet_base_model_results.txt"), 'w', encoding='utf-8') as f:
    f.write(f"Prophet 기본 모델 - {prophet_base_results.benchmarks}")


# 3. Change Point 민감도 분석 및 시각화
print("\n=== 3. Change Point 민감도 분석 ===")
cps_values = [0.01, 0.1, 0.5]
colors = ['red','yellow','blue']

results_cps = []
for cps, color in zip(cps_values,colors):
    plt.figure(figsize=(15,6))
    model_cps = Prophet(
        changepoint_prior_scale=cps,
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model_cps.fit(train[['ds', 'y']])

    if test[['ds']].empty:
        print(f"WARNING: Test data for Prophet CPS model (cps={cps}) is empty. Skipping prediction.")
        forecast_cps = pd.DataFrame()
    else:
        forecast_cps = model_cps.predict(test[['ds']])

    # Ensure forecast_cps has 'yhat' and is not empty before proceeding
    if 'yhat' not in forecast_cps.columns or forecast_cps.empty:
        print(f"ERROR: Prophet forecast_cps for cps={cps} does not contain 'yhat' column or is empty. Skipping plot for this CPS.")
        plt.close() # Close the current figure to prevent empty plot
        continue # Skip this iteration if forecast failed

    y_pred_cps_log = forecast_cps['yhat'].values
    y_pred_cps_orig = np.exp(y_pred_cps_log)

    mae_cps = mean_absolute_error(np.exp(test['y'].values), y_pred_cps_orig)
    mape_cps = MAPE_orig_scale(np.exp(test['y'].values), y_pred_cps_orig)
    results_cps.append({'changepoint_prior_scale': cps, 'MAE': mae_cps, 'MAPE': mape_cps})

    # Plot changepoints over actual data
    changepoints = model_cps.changepoints
    cp_df = pd.DataFrame({'ds': changepoints})
    # Check if cp_df is empty, if so, skip plotting changepoints
    if not cp_df.empty:
        trend_cp = model_cps.predict(cp_df)['trend']
        plt.scatter(changepoints, np.exp(trend_cp), label=f'Changepoints cps={cps:.2f}', color=color, s=25, zorder=5)
    else:
        print(f"Warning: No changepoints found for cps={cps}.")


    plt.plot(df_prophet['ds'], np.exp(df_prophet['y']), label='Actual', color='black', alpha=0.6, zorder=1)
    plt.legend()
    plt.title(f"Actual Data & Prophet Changepoints by changepoint_prior_scale, cps={cps:.2f}")
    plt.savefig(os.path.join(output_dir, f"prophet_changepoint_sensitivity_cps_{cps:.2f}.png"))
    plt.show()

# Separate plot for all changepoints combined
plt.figure(figsize=(15,10))
for cps, color in zip(cps_values,colors):
    model_cps = Prophet(
        changepoint_prior_scale=cps,
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model_cps.fit(train[['ds', 'y']])
    # Plot changepoints over actual data
    changepoints = model_cps.changepoints
    cp_df = pd.DataFrame({'ds': changepoints})
    if not cp_df.empty:
        trend_cp = model_cps.predict(cp_df)['trend']
        plt.scatter(changepoints, np.exp(trend_cp), label=f'Changepoints cps={cps}', s=25, zorder=5, color=color)
    else:
        print(f"Warning: No changepoints found for combined plot for cps={cps}.")

plt.plot(df_prophet['ds'], np.exp(df_prophet['y']), label='Actual', color='black', alpha=0.6, zorder=1)
plt.legend()
plt.title(f"Actual Data & Prophet Changepoints by changepoint_prior_scale")
plt.savefig(os.path.join(output_dir, f"prophet_changepoint_sensitivity.png"))
plt.show()


# 4. NeuralProphet 모델 적합 및 평가 (AR-Net + Trend+Seasonality+Holiday)
print("\n=== 4. NeuralProphet 모델 적합 및 평가 ===")

neuralprophet_results = ResponseVariable(title="NeuralProphet Model")

np_model = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    n_changepoints=50,
    changepoints_range=0.9,
    seasonality_mode='additive',
    epochs=100,
    batch_size=64,
    normalize='minmax',
    learning_rate=1.0
)

np_model = np_model.add_country_holidays('US')

train_np = train[['ds','y']].copy()
test_np = test[['ds','y']].copy()

# Ensure train_np and test_np are not empty before fitting/predicting
if train_np.empty:
    print("ERROR: Training data for NeuralProphet is empty. Cannot fit model. Exiting.")
    sys.exit(1)
if test_np.empty:
    print("WARNING: Test data for NeuralProphet is empty. Skipping prediction.")
    forecast_np = pd.DataFrame() # Create empty DataFrame to avoid errors
else:
    metrics_np = np_model.fit(train_np, freq='D')
    forecast_np = np_model.predict(test_np)

# Ensure forecast_np has 'yhat1' and is not empty before proceeding
if 'yhat1' not in forecast_np.columns or forecast_np.empty:
    print("ERROR: NeuralProphet forecast_np does not contain 'yhat1' column or is empty. Model prediction might have failed. Exiting.")
    sys.exit(1)


neuralprophet_results.past = df_all['daily_consumption_MW'].iloc[:split_idx]
neuralprophet_results.true = df_all['daily_consumption_MW'].iloc[split_idx:]
neuralprophet_results.pred = pd.Series(np.exp(forecast_np['yhat1'].values), index=test['ds'])
neuralprophet_results.forecast_index = test['ds']

neuralprophet_results.calculate_benchmarks(
    y_true_orig_scale=neuralprophet_results.true.values,
    y_pred_orig_scale=neuralprophet_results.pred.values,
    y_true_log_scale=test['y'].values,
    y_pred_log_scale=forecast_np['yhat1'].values
)

print(f"NeuralProphet 모델 - {neuralprophet_results.benchmarks}")
with open(os.path.join(output_dir, "neuralprophet_model_results.txt"), 'w', encoding='utf-8') as f:
    f.write(f"NeuralProphet 모델 - {neuralprophet_results.benchmarks}")


# 5. Future Regressors 분석 (Prophet)
print("\n=== 5. Future Regressors 포함 Prophet 모델 적합 및 평가 ===")

prophet_fr_results = ResponseVariable(title="Prophet + Future Regressors Model")

model_fr = Prophet(
    holidays=holidays,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model_fr.add_regressor('temp')
train_fr = train[['ds','y','temp']].copy()
test_fr = test[['ds','y','temp']].copy()

if train_fr.empty:
    print("ERROR: Training data for Prophet FR model is empty. Cannot fit model. Exiting.")
    sys.exit(1)
model_fr.fit(train_fr)

if test_fr[['ds','temp']].empty:
    print("WARNING: Test data for Prophet FR model is empty. Skipping prediction.")
    forecast_fr = pd.DataFrame()
else:
    forecast_fr = model_fr.predict(test_fr[['ds','temp']])

# Ensure forecast_fr has 'yhat' and is not empty before proceeding
if 'yhat' not in forecast_fr.columns or forecast_fr.empty:
    print("ERROR: Prophet forecast_fr does not contain 'yhat' column or is empty. Model prediction might have failed. Exiting.")
    sys.exit(1)

prophet_fr_results.past = df_all['daily_consumption_MW'].iloc[:split_idx]
prophet_fr_results.true = df_all['daily_consumption_MW'].iloc[split_idx:]
prophet_fr_results.pred = pd.Series(np.exp(forecast_fr['yhat'].values), index=test['ds'])
prophet_fr_results.forecast_index = test['ds']

prophet_fr_results.calculate_benchmarks(
    y_true_orig_scale=prophet_fr_results.true.values,
    y_pred_orig_scale=prophet_fr_results.pred.values,
    y_true_log_scale=test['y'].values,
    y_pred_log_scale=forecast_fr['yhat'].values
)


print(f"Future Regressors 포함 Prophet - {prophet_fr_results.benchmarks} (기본 모델 대비 MAE: {prophet_fr_results.benchmarks.mae - prophet_base_results.benchmarks.mae:+.3f}, MAPE: {prophet_fr_results.benchmarks.mape - prophet_base_results.benchmarks.mape:+.2f}%)")
with open(os.path.join(output_dir, "prophet_future_regressors_results.txt"), 'w', encoding='utf-8') as f:
    f.write(f"Future Regressors 포함 Prophet - {prophet_fr_results.benchmarks} (기본 모델 대비 MAE: {prophet_fr_results.benchmarks.mae - prophet_base_results.benchmarks.mae:+.3f}, MAPE: {prophet_fr_results.benchmarks.mape - prophet_base_results.benchmarks.mape:+.2f}%)")


# 6. Probabilistic Forecasting 및 이상징후 감지 (NeuralProphet quantile 사용)
print("\n=== 6. Probabilistic Forecasting 및 이상징후 감지 ===")

neuralprophet_prob_results = ResponseVariable(title="NeuralProphet Probabilistic Model")

np_model_mcmc = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    n_changepoints=50,
    changepoints_range=0.9,
    seasonality_mode='additive',
    epochs=100,
    batch_size=64,
    normalize='minmax',
    quantiles=[0.05, 0.95], # These define a 90% prediction interval
    learning_rate=1.0
)

np_model_mcmc = np_model_mcmc.add_country_holidays('US')

if train_np.empty:
    print("ERROR: Training data for NeuralProphet Probabilistic model is empty. Cannot fit model. Exiting.")
    sys.exit(1)
if test_np.empty:
    print("WARNING: Test data for NeuralProphet Probabilistic model is empty. Skipping prediction.")
    forecast_mcmc = pd.DataFrame()
else:
    metrics_mcmc = np_model_mcmc.fit(train_np, freq='D', continue_training=False)
    forecast_mcmc = np_model_mcmc.predict(test_np)

print("예측 결과 컬럼:", forecast_mcmc.columns.tolist())

# Ensure forecast_mcmc has required columns and is not empty before proceeding
required_prob_cols = ['yhat1', 'yhat1 5.0%', 'yhat1 95.0%']
if not all(col in forecast_mcmc.columns for col in required_prob_cols) or forecast_mcmc.empty:
    print(f"ERROR: NeuralProphet probabilistic forecast_mcmc is missing required columns ({required_prob_cols}) or is empty. Model prediction might have failed. Exiting.")
    sys.exit(1)


neuralprophet_prob_results.past = df_all['daily_consumption_MW'].iloc[:split_idx]
neuralprophet_prob_results.true = df_all['daily_consumption_MW'].iloc[split_idx:]
neuralprophet_prob_results.pred = pd.Series(np.exp(forecast_mcmc['yhat1'].values), index=test['ds'])
neuralprophet_prob_results.forecast_index = test['ds']

# Get lower and upper bounds on original scale for CRPS and outlier detection
y_pred_lower_log = forecast_mcmc['yhat1 5.0%'].values
y_pred_upper_log = forecast_mcmc['yhat1 95.0%'].values
lower_orig_scale = np.exp(y_pred_lower_log)
upper_orig_scale = np.exp(y_pred_upper_log)

# Calculate benchmarks for probabilistic model
neuralprophet_prob_results.calculate_benchmarks(
    y_true_orig_scale=neuralprophet_prob_results.true.values,
    y_pred_orig_scale=neuralprophet_prob_results.pred.values,
    y_true_log_scale=test['y'].values,
    y_pred_log_scale=forecast_mcmc['yhat1'].values,
    lower_bound=lower_orig_scale,
    upper_bound=upper_orig_scale
)

# Outlier detection
actual_orig_scale = np.exp(test['y'].values)
outliers = test['ds'][(actual_orig_scale < lower_orig_scale) | (actual_orig_scale > upper_orig_scale)]

probabilistic_results_str = f"평균 CRPS: {neuralprophet_prob_results.benchmarks.crps:.3f}\n"
probabilistic_results_str += f"95% 예측구간 벗어난 이상징후 발견 건수: {len(outliers)}\n"
probabilistic_results_str += "이상징후 날짜 예시:\n" + outliers.head().to_string()
print(probabilistic_results_str)
with open(os.path.join(output_dir, "neuralprophet_probabilistic_results.txt"), 'w', encoding='utf-8') as f:
    f.write(probabilistic_results_str)


# 7. 시각화 및 성능 지표 표
print("\n=== 7. 시각화 및 성능 지표 표 ===")

# 7-1) 기본 모델 vs Future Regressor vs NeuralProphet 예측 비교 그래프
# Before plotting, perform final checks on the data being plotted
plot_data_valid = True
if neuralprophet_prob_results.forecast_index is None or neuralprophet_prob_results.forecast_index.empty:
    print("Plotting Error: neuralprophet_prob_results.forecast_index is empty or None.")
    plot_data_valid = False
if neuralprophet_prob_results.true is None or neuralprophet_prob_results.true.empty:
    print("Plotting Error: neuralprophet_prob_results.true is empty or None.")
    plot_data_valid = False
if prophet_base_results.pred is None or prophet_base_results.pred.empty:
    print("Plotting Error: prophet_base_results.pred is empty or None.")
    plot_data_valid = False
if prophet_fr_results.pred is None or prophet_fr_results.pred.empty:
    print("Plotting Error: prophet_fr_results.pred is empty or None.")
    plot_data_valid = False
if neuralprophet_results.pred is None or neuralprophet_results.pred.empty:
    print("Plotting Error: neuralprophet_results.pred is empty or None.")
    plot_data_valid = False

if plot_data_valid:
    plt.figure(figsize=(15,6))
    plt.plot(neuralprophet_prob_results.forecast_index, neuralprophet_prob_results.true, label='Actual', color='black')
    plt.plot(prophet_base_results.forecast_index, prophet_base_results.pred, label='Prophet 기본 모델')
    plt.plot(prophet_fr_results.forecast_index, prophet_fr_results.pred, label='Prophet + Future Regressors')
    plt.plot(neuralprophet_results.forecast_index, neuralprophet_results.pred, label='NeuralProphet')
    
    # Check if confidence interval bounds are valid before plotting
    if 'yhat1 5.0%' in forecast_mcmc.columns and 'yhat1 95.0%' in forecast_mcmc.columns and \
       not lower_orig_scale.size == 0 and not upper_orig_scale.size == 0: # Check if arrays are not empty
        plt.fill_between(neuralprophet_prob_results.forecast_index, lower_orig_scale, upper_orig_scale, color='gray', alpha=0.3, label='NeuralProphet 95% CI')
    else:
        print("Warning: NeuralProphet 95% CI bounds not available or empty for plotting.")
        
    plt.legend()
    plt.title("테스트셋 예측 결과 비교")
    plt.savefig(os.path.join(output_dir, "test_set_prediction_comparison.png"))
    plt.show()
else:
    print("Skipping plotting due to missing or empty data for one or more series.")


# 7-2) 성능 지표 표 출력
# Initialize values with NaN, then populate if valid data exists
mae_values = [np.nan] * 4
rmse_values = [np.nan] * 4
mape_values = [np.nan] * 4
smape_values = [np.nan] * 4
crps_values = [np.nan] * 4

# Helper function to safely get benchmark value
def get_benchmark(result_obj, metric_attr):
    if result_obj and result_obj.benchmarks and getattr(result_obj.benchmarks, metric_attr) is not None:
        return getattr(result_obj.benchmarks, metric_attr)
    return np.nan

mae_values[0] = get_benchmark(prophet_base_results, 'mae')
mae_values[1] = get_benchmark(prophet_fr_results, 'mae')
mae_values[2] = get_benchmark(neuralprophet_results, 'mae')
mae_values[3] = get_benchmark(neuralprophet_prob_results, 'mae')

rmse_values[0] = get_benchmark(prophet_base_results, 'rmse')
rmse_values[1] = get_benchmark(prophet_fr_results, 'rmse')
rmse_values[2] = get_benchmark(neuralprophet_results, 'rmse')
rmse_values[3] = get_benchmark(neuralprophet_prob_results, 'rmse')

mape_values[0] = get_benchmark(prophet_base_results, 'mape')
mape_values[1] = get_benchmark(prophet_fr_results, 'mape')
mape_values[2] = get_benchmark(neuralprophet_results, 'mape')
mape_values[3] = get_benchmark(neuralprophet_prob_results, 'mape')

smape_values[0] = get_benchmark(prophet_base_results, 'smape')
smape_values[1] = get_benchmark(prophet_fr_results, 'smape')
smape_values[2] = get_benchmark(neuralprophet_results, 'smape')
smape_values[3] = get_benchmark(neuralprophet_prob_results, 'smape')

crps_values[3] = get_benchmark(neuralprophet_prob_results, 'crps')


df_results = pd.DataFrame({
    'Model': [
        prophet_base_results.title,
        prophet_fr_results.title,
        neuralprophet_results.title,
        neuralprophet_prob_results.title
    ],
    'MAE': mae_values,
    'RMSE': rmse_values,
    'MAPE (%)': mape_values,
    'sMAPE (%)': smape_values,
    'CRPS': crps_values
})

print("\n성능 지표:")
print(df_results.to_string(index=False))
df_results.to_csv(os.path.join(output_dir, "performance_metrics.txt"), sep='\t', index=False)