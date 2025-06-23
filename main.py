import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# main.py
import sys, csv
import argparse
from scripts.models.response_variable import ResponseVariable
from scripts.statmodels.analytical import ar, ma, arima, garch, linear, var
from scripts.statmodels.rnn import lstm, gru
import matplotlib.pyplot as plt
import pandas as pd


def compare_models(p, d, q, maxlags, horizon, data_file, vars, ic):
    """
    Run ARIMA, AR, MA, VAR models, collect ResponseVariable results, and plot them together.
    Plots both restored and log-diff predictions.
    """
    models = []
    labels = []
    # ARIMA
    arima_result = arima(p, d, q, horizon, data_file, vars, ic)
    models.append(arima_result)
    labels.append("ARIMA")
    # AR
    ar_result = ar(p, horizon, data_file, vars, ic)
    models.append(ar_result)
    labels.append("AR")
    # MA
    ma_result = ma(q, horizon, data_file, vars, ic)
    models.append(ma_result)
    labels.append("MA")
    # VAR (only if at least 2 variables)
    if len(vars) >= 2:
        var_result = var(maxlags, horizon, data_file, vars, ic)
        models.append(var_result)
        labels.append("VAR")

    compare_var = vars[0]

    # --- Plot restored values ---
    plt.figure(figsize=(12, 6))
    actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.true and model.true[compare_var] is not None:
            actual = model.true[compare_var]
            forecast_index = model.forecast_index
            break
    if actual is not None and forecast_index is not None:
        plt.plot(forecast_index, actual, label="Actual", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.pred and model.pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.pred[compare_var], label=f"{label} Prediction", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting {compare_var} (Restored Scale)"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    def circled_number(n):
        if 1 <= n <= 20:
            return chr(9311 + n)
        return str(n)
    variables = [f"{circled_number(index+1)} diff_log_{str(variable)}" for index, variable in enumerate(vars)]
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()

    # --- Plot log-diff values ---
    plt.figure(figsize=(12, 6))
    diff_actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.diff_log_true and model.diff_log_true[compare_var] is not None:
            diff_actual = model.diff_log_true[compare_var]
            forecast_index = model.forecast_index
            break
    if diff_actual is not None and forecast_index is not None:
        plt.plot(forecast_index, diff_actual, label="Actual (log-diff)", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.diff_log_pred and model.diff_log_pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.diff_log_pred[compare_var], label=f"{label} Prediction (log-diff)", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting diff_log_{compare_var}"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"diff_log_{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()
def compare_var_arima_gru(p, d, q, maxlags, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, ic, optimizer, loss, seed):
    """
    Run VAR, ARIMA, and GRU models, collect ResponseVariable results, and plot them together.
    Plots both restored and log-diff predictions.
    """
    models = []
    labels = []
    # ARIMA
    arima_result = arima(p, d, q, horizon, data_file, vars, ic)
    models.append(arima_result)
    labels.append("ARIMA")
    # VAR (only if at least 2 variables)
    if len(vars) >= 2:
        var_result = var(maxlags, horizon, data_file, vars, ic)
        models.append(var_result)
        labels.append("VAR")
    # GRU (only if at least 2 variables)
    if len(vars) >= 2:
        gru_result = gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
        models.append(gru_result)
        labels.append("GRU")

    compare_var = vars[0]

    # --- Plot restored values ---
    plt.figure(figsize=(12, 6))
    actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.true and model.true[compare_var] is not None:
            actual = model.true[compare_var]
            forecast_index = model.forecast_index
            break
    if actual is not None and forecast_index is not None:
        plt.plot(forecast_index, actual, label="Actual", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.pred and model.pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.pred[compare_var], label=f"{label} Prediction", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting {compare_var} (Restored Scale)"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    def circled_number(n):
        if 1 <= n <= 20:
            return chr(9311 + n)
        return str(n)
    variables = [f"{circled_number(index+1)} diff_log_{str(variable)}" for index, variable in enumerate(vars)]
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()

    # --- Plot log-diff values ---
    plt.figure(figsize=(12, 6))
    diff_actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.diff_log_true and model.diff_log_true[compare_var] is not None:
            diff_actual = model.diff_log_true[compare_var]
            forecast_index = model.forecast_index
            break
    if diff_actual is not None and forecast_index is not None:
        plt.plot(forecast_index, diff_actual, label="Actual (log-diff)", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.diff_log_pred and model.diff_log_pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.diff_log_pred[compare_var], label=f"{label} Prediction (log-diff)", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting diff_log_{compare_var}"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"diff_log_{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()
def compare_var_lstm_gru(p, d, q, maxlags, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, ic, optimizer, loss, seed):
    """
    Run VAR, LSTM, and GRU models, collect ResponseVariable results, and plot them together.
    Plots both restored and log-diff predictions.
    """
    models = []
    labels = []
    # VAR (only if at least 2 variables)
    if len(vars) >= 2:
        var_result = var(maxlags, horizon, data_file, vars, ic)
        models.append(var_result)
        labels.append("VAR")
    # LSTM (only if at least 2 variables)
    if len(vars) >= 2:
        lstm_result = lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
        models.append(lstm_result)
        labels.append("LSTM")
    # GRU (only if at least 2 variables)
    if len(vars) >= 2:
        gru_result = gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
        models.append(gru_result)
        labels.append("GRU")

    compare_var = vars[0]

    # --- Plot restored values ---
    plt.figure(figsize=(12, 6))
    actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.true and model.true[compare_var] is not None:
            actual = model.true[compare_var]
            forecast_index = model.forecast_index
            break
    if actual is not None and forecast_index is not None:
        plt.plot(forecast_index, actual, label="Actual", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.pred and model.pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.pred[compare_var], label=f"{label} Prediction", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting {compare_var} (Restored Scale)"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    def circled_number(n):
        if 1 <= n <= 20:
            return chr(9311 + n)
        return str(n)
    variables = [f"{circled_number(index+1)} diff_log_{str(variable)}" for index, variable in enumerate(vars)]
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()

    # --- Plot log-diff values ---
    plt.figure(figsize=(12, 6))
    diff_actual = None
    forecast_index = None
    for model in models:
        if compare_var in model.diff_log_true and model.diff_log_true[compare_var] is not None:
            diff_actual = model.diff_log_true[compare_var]
            forecast_index = model.forecast_index
            break
    if diff_actual is not None and forecast_index is not None:
        plt.plot(forecast_index, diff_actual, label="Actual (log-diff)", marker='o', color='black', linewidth=2, alpha=0.7)
    for model, label in zip(models, labels):
        if compare_var in model.diff_log_pred and model.diff_log_pred[compare_var] is not None:
            plt.plot(model.forecast_index, model.diff_log_pred[compare_var], label=f"{label} Prediction (log-diff)", marker='x', linewidth=2)
    title = f"Forecast Comparison - Predicting diff_log_{compare_var}"
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"diff_log_{compare_var}", fontsize=14)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', color='gray', alpha=0.4)
    plt.xticks(rotation=30, fontsize=11)
    plt.yticks(fontsize=11)
    notes = f"{len(variables)} variables used:\n{chr(10).join(variables)}"
    plt.annotate(
        notes,
        xy=(0.01, 0.99), xycoords='axes fraction',
        fontsize=12, color="black",
        xytext=(0, 0), textcoords='offset points',
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc="white", ec="black", alpha=0.3)
    )
    plt.tight_layout()
    plt.show()
def main():
    parser = argparse.ArgumentParser(description=f"Run an model for time series forecasting.") 
    # args
    parser.add_argument("activity", help="The name of the activity (e.g., casual, forecast)")
    parser.add_argument("model_name", help="The type of the model (e.g., lstm, gru).")
    # kwargs
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility.")
    parser.add_argument("--horizon", type=int, default=12, help="Number of sdteps to forecast (Horizon).")
    parser.add_argument("--lag", type=int, default=12, help="(VAR) Sequence length")
    parser.add_argument("--p", type=int, default=-1, help="(AR, ARIMA, GARCH) Number of lagged autoregressive terms")
    parser.add_argument("--d", type=int, default=-1, help="(ARIMA) Number of nonseasonal differences")
    parser.add_argument("--q", type=int, default=-1, help="(MA, ARIMA, GARCH) Number of lagged moving average terms")
    parser.add_argument("--maxlags", type=int, default=15, help="(VAR) Maximum number of lags to consider.")
    parser.add_argument("--neurons", type=int, default=200, help="(RNN) Number of neurons in the layer.") 
    parser.add_argument("--layers", type=int, default=1, help="(RNN) Number of layers in the for Deep Learning.") 
    parser.add_argument("--batch_size", type=int, default=16, help="(RNN) Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="(RNN) Number of training epochs.")
    parser.add_argument("--data_file", type=str, default="assets/data.csv", help="Path to the data file (JSON).")
    parser.add_argument("--vars", type=str, default="cpi,ppi", help="Comma-separated list of variables (e.g., 'cpi,ppi,inflation')")
    parser.add_argument("--ic", type=str, default="aic", help="(VAR) Information Criterion such as AIC, BIC")
    parser.add_argument("--optimizer", type=str, default="adam", help="(VAR) Optimizer")
    parser.add_argument("--loss", type=str, default="mean_squared_error", help="(VAR) Loss Function.")
    parser.add_argument("--compare_models", action="store_true", help="Compare ARIMA, AR, MA, VAR models in a single plot.")
    parser.add_argument("--compare_var_arima_gru", action="store_true", help="Compare VAR, ARIMA, GRU models in a single plot.")
    parser.add_argument("--compare_var_lstm_gru", action="store_true", help="Compare VAR, LSTM, GRU models in a single plot.")
    args = parser.parse_args()
    
    # args
    activity = args.activity
    model_name = args.model_name

    # kwargs
    seed = args.seed
    horizon = args.horizon
    lag = args.lag
    p = args.p
    d = args.d
    q = args.q
    maxlags = args.maxlags
    neurons = args.neurons
    layers = args.layers
    batch_size = args.batch_size
    epochs = args.epochs
    data_file = args.data_file
    ic = args.ic
    optimizer = args.optimizer
    loss = args.loss
    # Parse vars as list
    vars = [v.strip() for v in args.vars.split(",") if v.strip()]
    if len(vars) < 1:
        print("Error: At least one variable must be specified with --vars")
        sys.exit(1)

    if args.compare_var_lstm_gru or (hasattr(args, "activity") and args.activity == "compare_var_lstm_gru"):
        compare_var_lstm_gru(
            p, d, q, maxlags, horizon, lag, neurons, layers, epochs, batch_size,
            data_file, vars, ic, optimizer, loss, seed
        )
        return
    if args.compare_var_arima_gru or (hasattr(args, "activity") and args.activity == "compare_var_arima_gru"):
        compare_var_arima_gru(
            p, d, q, maxlags, horizon, lag, neurons, layers, epochs, batch_size,
            data_file, vars, ic, optimizer, loss, seed
        )
        return
    if activity == 'compare_rnn':
        compare_var_lstm_gru(
            p, d, q, maxlags, horizon, lag, neurons, layers, epochs, batch_size,
            data_file, vars, ic, optimizer, loss, seed
        )
        return

    if activity == 'compare':
        compare_models(p, d, q, maxlags, horizon, data_file, vars, ic)
        return

    elif activity == 'forecast':
        print(f"Running {model_name} model with the following parameters:")
        if model_name == 'arima':
            result = arima(p, d, q, horizon, data_file, vars, ic)
        elif model_name == 'ar':
            result = ar(p, horizon, data_file, vars, ic)
        elif model_name == 'ma':
            result = ma(q, horizon, data_file, vars, ic)
        elif model_name == 'garch':
            result = garch(p, q, horizon, data_file, vars)
        elif model_name == 'linear':
            if len(vars) < 2:
                print("Error: At least two variables required for linear model.")
                sys.exit(1)
            result = linear(horizon, data_file, vars)
        elif model_name == 'var':
            if len(vars) < 2:
                print("Error: At least two variables required for VAR model.")
                sys.exit(1)
            print(f"Running VAR model with maxlags={maxlags}, ic={ic}")
            result = var(maxlags, horizon, data_file, vars, ic)
        elif model_name == 'lstm':
            if len(vars) < 2:
                print("Error: At least two variables required for LSTM model.")
                sys.exit(1)
            result = lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
        elif model_name == 'gru':
            if len(vars) < 2:
                print("Error: At least two variables required for GRU model.")
                sys.exit(1)
            result = gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
        # Evaluation
        print('\n----------Benchmark----------\n')
        print(f"Model: {model_name}, Seed: {seed}, Horizon: {horizon}, Variables: {', '.join(vars)}")
        result.bench()
        for bm in result.benchmarks:
            print(bm.__repr__())
        # Plot
        if (not layers == 1) and (model_name== 'lstm' or model_name == 'gru'): model_name += f" layer{layers}" 

        result.plot(True,True)
        result.plot(True,False)
        result.plot(False,False)
        result.plot(False,True)

    elif activity == 'generator':
        # ARIMA
        dicts = []
        result = arima(p, d, q, horizon, data_file, vars, ic)
        result.bench()
        dic = result.benchmarks.to_dict()
        dicts.append(dic)
        header = dicts[0].keys()
        with open(f'results/forecast/csv/arima, {",".join(vars)}.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(dicts)
        # linear
        dicts = []
        if len(vars) < 2:
            print("Error: At least two variables required for linear model.")
            sys.exit(1)
        result = linear(horizon, data_file, vars)
        result.bench()
        dic = result.benchmarks.to_dict()
        dicts.append(dic)
        header = dicts[0].keys()
        with open(f'results/forecast/csv/linear, {",".join(vars)}.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(dicts)

        # VAR 
        dicts = []
        if len(vars) < 2:
            print("Error: At least two variables required for VAR model.")
            sys.exit(1)
        result = var(maxlags, horizon, data_file, vars, ic)
        result.bench()
        dic = result.benchmarks.to_dict()
        dicts.append(dic)
        header = dicts[0].keys()
        with open(f'results/forecast/csv/VAR, {",".join(vars)}.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(dicts)
        dicts=[]

        # LSTM
        if len(vars) < 2:
            print("Error: At least two variables required for LSTM model.")
            sys.exit(1)
        for seed in range(1,11): 
            result = lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
            result.bench()
            dic = result.benchmarks.to_dict()
            dicts.append(dic)

        header = dicts[0].keys()
        with open(f'results/forecast/csv/LSTM, {",".join(vars)}.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(dicts)

        # GRU
        if len(vars) < 2:
            print("Error: At least two variables required for GRU model.")
            sys.exit(1)
        for seed in range(1,11): 
            result = gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
            result.bench()
            dic = result.benchmarks.to_dict()
            dicts.append(dic)

        header = dicts[0].keys()
        with open(f'results/forecast/csv/GRU, {",".join(vars)}.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(dicts)
          
    else: 
        print("Error: The first argument should be 'casual' or 'forecast'")
        print("Example: python run.py forecast gru --vars 'ppi,inflation'")
        sys.exit(1)

if __name__ == "__main__":
    main()