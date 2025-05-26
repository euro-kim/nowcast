import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import os

# Create a directory to save results if it doesn't exist
output_dir = "results/prob2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv("MACRO.csv", parse_dates=["observation_date"])
df = df[(df['observation_date'] >= '1990-01') & (df['observation_date'] <= '2024-12')]
df.set_index('observation_date', inplace=True)

# 로그 및 1차 차분
df['log_GDPC1'] = np.log(df['GDPC1'])
df['log_CPI'] = np.log(df['CPIAUCSL'])

df['dlog_GDPC1'] = df['log_GDPC1'].diff()
df['dlog_CPI'] = df['log_CPI'].diff()
df['dFFR'] = df['FEDFUNDS'].diff()

# 결측치 제거 (처음 한 개 행은 NaN)
df.dropna(inplace=True)

# Z-score 표준화
df['z_GDP'] = zscore(df['dlog_GDPC1'])
df['z_CPI'] = zscore(df['dlog_CPI'])
df['z_FFR'] = zscore(df['dFFR'])

# 2. Taylor Rule 기반 금리 예측 및 통화정책 충격 정의
# FFR_hat = α + β1 * Inflation + β2 * Output_gap + ε
X = sm.add_constant(df[['z_CPI', 'z_GDP']])
y = df['z_FFR']
model = sm.OLS(y, X).fit()
df['FFR_hat'] = model.predict(X)
df['MP_shock'] = df['z_FFR'] - df['FFR_hat']

# 3. Local Projection 함수 정의
def local_projection(df, shock_col, target_col, max_h=24):
    results = []
    for h in range(0, max_h + 1):
        df[f'{target_col}_lead'] = df[target_col].shift(-h)
        temp = df[[f'{target_col}_lead', shock_col]].dropna()
        X = sm.add_constant(temp[shock_col])
        y = temp[f'{target_col}_lead']
        res = sm.OLS(y, X).fit()
        results.append((h, res.params[1], res.bse[1]))
    return pd.DataFrame(results, columns=['horizon', 'IRF', 'StdErr'])

irf_gdp = local_projection(df.copy(), 'MP_shock', 'z_GDP')
irf_cpi = local_projection(df.copy(), 'MP_shock', 'z_CPI')

# 4. 경기 국면 더미 포함 조건부 Local Projection
# 침체 구분: GDP 성장률이 평균 이하이면 침체로 간주
recession_dummy = (df['z_GDP'] < df['z_GDP'].mean()).astype(int)
df['recession'] = recession_dummy

def conditional_lp(df, shock_col, target_col, condition_col, max_h=24):
    results = []
    for h in range(0, max_h + 1):
        df[f'{target_col}_lead'] = df[target_col].shift(-h)
        temp = df[[f'{target_col}_lead', shock_col, condition_col]].dropna()
        temp['interaction'] = temp[shock_col] * temp[condition_col]
        X = sm.add_constant(temp[[shock_col, 'interaction']])
        y = temp[f'{target_col}_lead']
        res = sm.OLS(y, X).fit()
        results.append((h, res.params[1], res.params[2]))
    return pd.DataFrame(results, columns=['horizon', 'Shock_IRF', 'Interaction_IRF'])

cond_irf_gdp = conditional_lp(df.copy(), 'MP_shock', 'z_GDP', 'recession')
cond_irf_cpi = conditional_lp(df.copy(), 'MP_shock', 'z_CPI', 'recession')

# 5. IRF 그래프 저장
plt.figure(figsize=(10, 5))
plt.plot(irf_gdp['horizon'], irf_gdp['IRF'], label='GDP IRF')
plt.fill_between(irf_gdp['horizon'], irf_gdp['IRF'] - 1.96 * irf_gdp['StdErr'],
                 irf_gdp['IRF'] + 1.96 * irf_gdp['StdErr'], alpha=0.3)
plt.title('IRF of GDP to MP Shock (Local Projection)')
plt.xlabel('Months')
plt.ylabel('Response')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "IRF_GDP.png"))
plt.close() # Close the plot to free up memory

plt.figure(figsize=(10, 5))
plt.plot(irf_cpi['horizon'], irf_cpi['IRF'], label='CPI IRF', color='red')
plt.fill_between(irf_cpi['horizon'], irf_cpi['IRF'] - 1.96 * irf_cpi['StdErr'],
                 irf_cpi['IRF'] + 1.96 * irf_cpi['StdErr'], alpha=0.3, color='red')
plt.title('IRF of CPI to MP Shock (Local Projection)')
plt.xlabel('Months')
plt.ylabel('Response')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "IRF_CPI.png"))
plt.close() # Close the plot to free up memory

# 6. VAR 기반 IRF 저장
var_df = df[['z_GDP', 'z_CPI', 'z_FFR']]
var_model = VAR(var_df)
var_results = var_model.fit(6)

irf_var = var_results.irf(24)
fig = irf_var.plot(orth=False) # plot() returns a matplotlib Figure object
fig.savefig(os.path.join(output_dir, "IRF_VAR.png"))
plt.close(fig) # Close the figure to free up memory

# 7. 텍스트 결과 저장
with open(os.path.join(output_dir, "LP_VAR_IRF_Results.txt"), "w") as f:
    f.write("==== OLS (Taylor Rule) ====\n")
    f.write(str(model.summary()))
    f.write("\n\n==== Local Projection GDP IRF ====\n")
    f.write(irf_gdp.to_string(index=False))
    f.write("\n\n==== Local Projection CPI IRF ====\n")
    f.write(irf_cpi.to_string(index=False))
    f.write("\n\n==== Conditional LP GDP ====\n")
    f.write(cond_irf_gdp.to_string(index=False))
    f.write("\n\n==== Conditional LP CPI ====\n")
    f.write(cond_irf_cpi.to_string(index=False))
    f.write("\n\n==== VAR Summary ====\n")
    f.write(str(var_results.summary()))