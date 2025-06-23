import warnings, sys
import numpy as np
from numpy import array
import pandas as pd
from pandas import DataFrame, Series
from scripts.models.response_variable import ResponseVariable
from scripts.common.structure import init_df, diff_df
from scripts.common.statistics import adf_stats, pearson_correlations, descriptive_stats
from scripts.common.var import regression_summary, lag_order_selection, full_granger_causality
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

def create_sequences(data: np.ndarray, lag: int, input_cols_idx: list[int], target_col_idx: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i - lag:i, input_cols_idx])  # shape: (lag, num_features)
        y.append(data[i, target_col_idx])
    return np.array(X), np.array(y)

def ar(p: int, horizon: int, data_file: str, vars: list[str], ic: str) -> ResponseVariable:
    if len(vars) != 1:
        warnings.warn(f"AR model requires exactly one variable. Only the first variable will be used.", UserWarning)
        vars = vars[:1]
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'AR'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    for var in vars:
        series = df[var]
        log_series = df[f'log_{var}']
        diff_log_series = df[f'diff_log_{var}']

        result.set_series_segments(var, series, diff_log_series, horizon)

        train_data = diff_log_series[:-horizon].dropna()
        # Select best p if p == -1
        best_p = p
        if p == -1:
            best_ic = float("inf")
            best_p = 1
            for test_p in range(1, 6):
                try:
                    model = ARIMA(train_data, order=(test_p, 0, 0))
                    fitted = model.fit()
                    val = getattr(fitted, ic, None)
                    if val is not None and val < best_ic:
                        best_ic = val
                        best_p = test_p
                except Exception:
                    continue
        if len(train_data) < best_p:
            print(f"Error: Not enough training data for {var}. len(train_data)={len(train_data)}, p={best_p}")
            sys.exit(1)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model = ARIMA(train_data, order=(best_p, 0, 0))
                fitted_model = model.fit()
        except Exception as e:
            print(f"Error fitting AR model for {var}: {e}")
            sys.exit(1)
        result.model_name = f'AR({best_p})'
        forecast_index = diff_log_series.iloc[-horizon:].index
        result.diff_log_pred[var] = pd.Series(fitted_model.forecast(steps=horizon), index=forecast_index)
        last_log = log_series.iloc[-horizon - 1]
        forecast_log = result.diff_log_pred[var].cumsum() + last_log
        result.pred[var] = np.exp(forecast_log) - 1e-6

        # Save fitted model and parameters
        result.fitted[var] = fitted_model
        # Flatten parameters for AR
        params_dict = fitted_model.params.to_dict() if hasattr(fitted_model, "params") else {}
        result.params[var] = [
            {
                "variable": var,
                "model": "AR",
                "order": (best_p, 0, 0),
                "parameter": pname,
                "value": pval
            }
            for pname, pval in params_dict.items()
        ]
        result.save_params_csv(var)

    return result

def ma(q: int, horizon: int, data_file: str, vars: list[str], ic: str) -> ResponseVariable:
    if len(vars) != 1:
        warnings.warn(f"MA model requires exactly one variable. Only the first variable will be used.", UserWarning)
        vars = vars[:1]
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'MA'
    result.model_name = f'MA'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    for var in vars:
        series = df[var]
        log_series = df[f'log_{var}']
        diff_log_series = df[f'diff_log_{var}']

        result.set_series_segments(var, series, diff_log_series, horizon)

        train_data = diff_log_series[:-horizon].dropna()
        # Select best q if q == -1
        best_q = q
        if q == -1:
            best_ic = float("inf")
            best_q = 1
            for test_q in range(1, 6):
                try:
                    model = ARIMA(train_data, order=(0, 0, test_q))
                    fitted = model.fit()
                    val = getattr(fitted, ic, None)
                    if val is not None and val < best_ic:
                        best_ic = val
                        best_q = test_q
                except Exception:
                    continue
        if len(train_data) < best_q:
            print(f"Error: Not enough training data for {var}. len(train_data)={len(train_data)}, q={best_q}")
            sys.exit(1)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model = ARIMA(train_data, order=(0, 0, best_q))
                fitted_model = model.fit()
        except Exception as e:
            print(f"Error fitting MA model for {var}: {e}")
            sys.exit(1)
        
        result.model_name = f'MA({best_q})'

        forecast_index = diff_log_series.iloc[-horizon:].index
        result.diff_log_pred[var] = pd.Series(fitted_model.forecast(steps=horizon), index=forecast_index)
        last_log = log_series.iloc[-horizon - 1]
        forecast_log = result.diff_log_pred[var].cumsum() + last_log
        result.pred[var] = np.exp(forecast_log) - 1e-6

        # Save fitted model and parameters
        result.fitted[var] = fitted_model
        # Flatten parameters for MA
        params_dict = fitted_model.params.to_dict() if hasattr(fitted_model, "params") else {}
        result.params[var] = [
            {
                "variable": var,
                "model": "MA",
                "order": (0, 0, best_q),
                "parameter": pname,
                "value": pval
            }
            for pname, pval in params_dict.items()
        ]
        result.save_params_csv(var)

    return result

def arima(p: int, d: int, q: int, horizon: int, data_file: str, vars: list[str], ic: str = "aic") -> ResponseVariable:
    if len(vars) != 1:
        warnings.warn(f"ARIMA model requires exactly one variable. Only the first variable will be used.", UserWarning)
        vars = vars[:1]
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'ARIMA'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    for var in vars:
        series = df[var]
        log_series = df[f'log_{var}']
        diff_log_series = df[f'diff_log_{var}']

        result.set_series_segments(var, series, diff_log_series, horizon)

        train_data = diff_log_series[:-horizon].dropna()
        # Select best p, d, q if any is -1
        best_p, best_d, best_q = p, d, q
        if p == -1 or d == -1 or q == -1:
            best_ic = float("inf")
            best_combo = (1, 0, 1)
            for test_p in range(1, 6) if p == -1 else [p]:
                for test_d in range(0, 2) if d == -1 else [d]:
                    for test_q in range(1, 6) if q == -1 else [q]:
                        try:
                            model = ARIMA(train_data, order=(test_p, test_d, test_q))
                            fitted = model.fit()
                            val = getattr(fitted, ic, None)
                            if val is not None and val < best_ic:
                                best_ic = val
                                best_combo = (test_p, test_d, test_q)
                        except Exception:
                            continue
            best_p, best_d, best_q = best_combo
        if len(train_data) < max(best_p, best_q):
            print(f"Error: Not enough training data for {var}. len(train_data)={len(train_data)}, p={best_p}, q={best_q}")
            sys.exit(1)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                model = ARIMA(train_data, order=(best_p, best_d, best_q))
                fitted_model = model.fit()
        except Exception as e:
            print(f"Error fitting ARIMA model for {var}: {e}")
            sys.exit(1)

        result.model_name = f'ARIMA({best_p},{best_d},{best_q})'
        
        forecast_index = diff_log_series.iloc[-horizon:].index
        result.diff_log_pred[var] = pd.Series(fitted_model.forecast(steps=horizon), index=forecast_index)
        last_log = log_series.iloc[-horizon - 1]
        forecast_log = result.diff_log_pred[var].cumsum() + last_log
        result.pred[var] = np.exp(forecast_log) - 1e-6

        # Save fitted model and parameters
        result.fitted[var] = fitted_model
        # Flatten parameters for ARIMA
        params_dict = fitted_model.params.to_dict() if hasattr(fitted_model, "params") else {}
        result.params[var] = [
            {
                "variable": var,
                "model": "ARIMA",
                "order": (best_p, best_d, best_q),
                "parameter": pname,
                "value": pval
            }
            for pname, pval in params_dict.items()
        ]
        result.save_params_csv(var)

    return result

def garch(p: int, q: int, horizon: int, data_file: str, vars: list[str]) -> ResponseVariable:
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'GARCH'
    result.model_name = f'GARCH'
    
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    for var in vars:
        series = df[var]
        log_series = df[f'log_{var}']
        diff_log_series = df[f'diff_log_{var}']

        result.set_series_segments(var, series, diff_log_series, horizon)

        train_data = diff_log_series[:-horizon].dropna()
        if len(train_data) < max(p, q) + 1:
            print(f"Error: Not enough training data for {var}. len(train_data)={len(train_data)}, required={max(p, q) + 1}")
            sys.exit(1)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = arch_model(train_data, p=p, q=q, vol='GARCH', dist='normal')
                fitted_model = model.fit(disp='off')
        except Exception as e:
            print(f"Error fitting GARCH model for {var}: {e}")
            sys.exit(1)

        forecast_result = fitted_model.forecast(horizon=horizon)
        forecast_index = diff_log_series.iloc[-horizon:].index
        result.diff_log_pred[var] = pd.Series(forecast_result.mean.iloc[-1].values, index=forecast_index)
        last_log = log_series.iloc[-horizon - 1]
        forecast_log = result.diff_log_pred[var].cumsum() + last_log
        result.pred[var] = np.exp(forecast_log) - 1e-6

        # Save fitted model and parameters
        result.fitted[var] = fitted_model
        # Flatten parameters for GARCH
        params_dict = fitted_model.params.to_dict() if hasattr(fitted_model, "params") else {}
        result.params[var] = [
            {
                "variable": var,
                "model": "GARCH",
                "order": (p, q),
                "parameter": pname,
                "value": pval
            }
            for pname, pval in params_dict.items()
        ]
        result.save_params_csv(var)

    return result

def linear(horizon: int, data_file: str, vars: list[str]) -> ResponseVariable :
    df = init_df(data_file)
    df = diff_df(df, *vars)

    result = ResponseVariable()
    result.model_type = f'Linear'
    result.model_name = f'Linear'
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    feature_vars = [f'diff_log_{v}' for v in vars[1:]]
    X = df[feature_vars].values

    for idx, var in enumerate(vars):
        series = df[var]
        log_series = df[f'log_{var}']
        diff_log_series = df[f'diff_log_{var}']

        mask = ~np.isnan(X[:-horizon]).any(axis=1) & ~pd.isna(diff_log_series[:-horizon])
        train_X = X[:-horizon][mask]
        train_y = diff_log_series[:-horizon][mask]
        test_X = X[-horizon:]

        result.set_series_segments(var, series, diff_log_series, horizon)

        model = LinearRegression()
        model.fit(train_X, train_y)

        forecast_index = diff_log_series.iloc[-horizon:].index
        Array = model.predict(test_X)
        result.diff_log_pred[var] = pd.Series(Array, index=forecast_index)
        last_log = log_series.iloc[-horizon - 1]
        log_pred = np.cumsum(result.diff_log_pred[var]) + last_log
        result.pred[var] = np.exp(log_pred) - 1e-6

        # Save fitted model and parameters
        result.fitted[var] = model
        # Flatten parameters for LinearRegression
        coefs = model.coef_.tolist() if hasattr(model, "coef_") else []
        intercept = model.intercept_.item() if hasattr(model, "intercept_") else None
        result.params[var] = [
            {
                "variable": var,
                "model": "LinearRegression",
                "order": None,
                "parameter": f"coef_{feat}",
                "value": coef
            }
            for feat, coef in zip(feature_vars, coefs)
        ]
        result.params[var].append({
            "variable": var,
            "model": "LinearRegression",
            "order": None,
            "parameter": "intercept",
            "value": intercept
        })
        result.save_params_csv(var)

    return result

def var(maxlags: int, horizon: int, data_file: str, vars: list[str], ic: str) -> ResponseVariable:
    import logging
    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger("VAR_DEBUG")

    try:
        df = init_df(data_file)
        df = diff_df(df, *vars)
    except Exception as e:
        logger.error(f"Failed to load or preprocess data: {e}")
        return None

    try:
        descriptive_stats(df, vars)
        adf_stats(df, vars)
        pearson_correlations(df, vars)
    except Exception as e:
        logger.warning(f"Diagnostics failed: {e}")

    result = ResponseVariable()
    result.model_type = f'VAR'  
    result.forecast_index = df.index[-horizon:]
    result.set_variables(vars)

    try:
        diff_log_dataframe = df[[f'diff_log_{v}' for v in vars]]
        train_data = diff_log_dataframe[:-horizon].dropna()
        if len(train_data) < maxlags:
            logger.error(f"Not enough training data.  len(train_data)={len(train_data)}, maxlags={maxlags}")
            return None

        model = VAR(train_data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model_fitted = model.fit(maxlags=maxlags, ic=ic)
    except Exception as e:
        logger.error(f"Error fitting VAR model: {e}")
        return None

    try:
        regression_summary(model_fitted, vars)
        lag_order_selection(model, maxlags, vars)
        full_granger_causality(model_fitted, vars, maxlags)
    except Exception as e:
        logger.warning(f"Post-model diagnostics failed: {e}")

    try:
        lag_order = model_fitted.k_ar
        forecast_input = train_data.values[-lag_order:]
        forecast_output = model_fitted.forecast(y=forecast_input, steps=horizon)
    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        return None
    result.model_name = f'VAR({lag_order})'  

    try:
        for i, var in enumerate(vars):
            series = df[var]
            log_series = df[f'log_{var}']
            diff_log_series = df[f'diff_log_{var}']

            result.set_series_segments(var, series, diff_log_series, horizon)

            forecast_index = diff_log_series.iloc[-horizon:].index
            result.diff_log_pred[var] = pd.Series(forecast_output[:, i], index=forecast_index)
            last_log = log_series.iloc[-horizon - 1]
            log_pred = np.cumsum(result.diff_log_pred[var]) + last_log
            result.pred[var] = np.exp(log_pred) - 1e-6

            # Save fitted model and parameters
            result.fitted[var] = model_fitted
            # Flatten parameters for VAR
            coefs = model_fitted.coefs.tolist() if hasattr(model_fitted, "coefs") else None
            intercept = model_fitted.intercept.tolist() if hasattr(model_fitted, "intercept") else None
            variables_ = vars
            rows = []
            if coefs is not None:
                for lag_idx, lag_matrix in enumerate(coefs):
                    for i, target_var in enumerate(variables_):
                        if target_var != var:
                            continue
                        for j, source_var in enumerate(variables_):
                            rows.append({
                                "variable": target_var,
                                "model": "VAR",
                                "order": model_fitted.k_ar if hasattr(model_fitted, "k_ar") else None,
                                "parameter": f"coef_lag{lag_idx+1}_{source_var}",
                                "value": lag_matrix[i][j]
                            })
            if intercept is not None:
                for i, target_var in enumerate(variables_):
                    if target_var != var:
                        continue
                    rows.append({
                        "variable": target_var,
                        "model": "VAR",
                        "order": model_fitted.k_ar if hasattr(model_fitted, "k_ar") else None,
                        "parameter": "intercept",
                        "value": intercept[i] if isinstance(intercept, list) else intercept
                    })
            result.params[var] = rows
            result.save_params_csv(var)
            # Call Diebold-Mariano test for each variable (VAR)
            result.diebold_mariano_test(var)
    except Exception as e:
        logger.error(f"Error assembling forecast results: {e}")
        return None

    return result

