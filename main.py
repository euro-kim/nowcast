# main.py
import sys, csv
import argparse
from scripts.forecast import linear, var, lstm, gru, varbase
from scripts.common import evaluate_forecast, plot_forecast, plot_forecast_past


SCRIPT_MAPPING = {
    "var": "scripts/forecast/var.py",
    "linear": "scripts/forecast/linear.py",
    "gru": "scripts/forecast/gru.py",
    "lstm": "scripts/forecast/lstm.py",
}

def main():
    parser = argparse.ArgumentParser(description=f"Run an model for time series forecasting.") 
    # args
    parser.add_argument("activity", help="The name of the activity (e.g., casual, project)")
    parser.add_argument("model_name", help="The type of the model (e.g., lstm, transformer).")
    # kwargs
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility.")
    parser.add_argument("--horizon", type=int, default=12, help="Number of steps to forecast (Horizon).")
    parser.add_argument("--lag", type=int, default=12, help="Sequence length (lag).")
    parser.add_argument("--maxlags", type=int, default=15, help="Maximum number of lags to consider.")
    parser.add_argument("--neurons", type=int, default=200, help="Number of neurons in the LSTM layer.") 
    parser.add_argument("--layers", type=int, default=1, help="Number of layers in the for Deep Learning.") 
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--data_file", type=str, default="assets/data.json", help="Path to the data file (JSON).")
    parser.add_argument("--var0", type=str, default="cpi", help="Response Variable")
    parser.add_argument("--var1", type=str, default="ppi", help="Explanatory Variable")
    parser.add_argument("--ic", type=str, default="aic", help="Information Criterion such as AIC, BIC")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--loss", type=str, default="mean_squared_error", help="Loss Function.")
    
    args = parser.parse_args()
    
    # args
    acivity = args.activity
    model_name = args.model_name

    # kwargs
    seed = args.seed
    horizon = args.horizon
    lag = args.lag
    maxlags = args.maxlags
    neurons = args.neurons
    layers = args.layers
    batch_size = args.batch_size
    epochs = args.epochs
    data_file = args.data_file
    var0 = args.var0
    var1 = args.var1
    ic = args.ic
    optimizer = args.optimizer
    loss = args.loss
    if acivity == 'base':
        y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log = varbase(seed, maxlags, horizon, data_file, var0, ic)
        # Evaluation
        rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
        rmse, mae, r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)

    if acivity == 'forecast':
        if model_name == 'linear':
            y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log = linear(seed, horizon, data_file, var0, var1)
        if model_name == 'var':
            y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log = var(seed, maxlags, horizon, data_file, var0, var1, ic)
        if model_name == 'lstm':
            y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log = lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss)
        if model_name == 'gru':
            y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log = gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss)
        # Evaluation
        rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
        rmse, mae, r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)

        # Plot
        if (not layers == 1) and (model_name== 'lstm' or model_name == 'gru'): model_name += f" layer{layers}" 
        plot_forecast(False, y_true, y_pred, var0, var1, forecast_index, title_suffix=f'{model_name}') 
        plot_forecast(True, y_true_diff_log, y_pred_diff_log, var0, var1, forecast_index, title_suffix=f'{model_name}') 
        plot_forecast_past(False, y_past ,y_true, y_pred, var0, var1, forecast_index, title_suffix=f'{model_name}')
    if acivity == 'generator':
            dicts=[]
            for seed in range(1,11): 
                y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log= linear(seed, horizon, data_file, var0, var1)
                rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
                diff_log_rmse, diff_log_mae, diff_log_r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)
                dic={
                     'rmse': rmse,
                     'mae': mae,
                     'r2': r2,
                     'diff_log_rmse': diff_log_rmse,
                     'diff_log_mae': diff_log_mae,
                     'diff_log_r2': diff_log_r2,
                }
                dicts.append(dic)
            header = dicts[0].keys()
            with open(f'results/forecast/csv/linear, {var0}, {var1}.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(dicts)
            dicts=[]
            for seed in range(1,11): 
                y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log= var(seed, maxlags, horizon, data_file, var0, var1, ic)
                rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
                diff_log_rmse, diff_log_mae, diff_log_r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)
                dic={
                     'rmse': rmse,
                     'mae': mae,
                     'r2': r2,
                     'diff_log_rmse': diff_log_rmse,
                     'diff_log_mae': diff_log_mae,
                     'diff_log_r2': diff_log_r2,
                }
                dicts.append(dic)
            header = dicts[0].keys()
            with open(f'results/forecast/csv/var, {var0}, {var1}.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(dicts)
            dicts=[]
            for seed in range(1,11): 
                y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log= lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss)
                rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
                diff_log_rmse, diff_log_mae, diff_log_r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)
                dic={
                     'rmse': rmse,
                     'mae': mae,
                     'r2': r2,
                     'diff_log_rmse': diff_log_rmse,
                     'diff_log_mae': diff_log_mae,
                     'diff_log_r2': diff_log_r2,
                }
                dicts.append(dic)
            header = dicts[0].keys()
            with open(f'results/forecast/csv/lstm, {var0}, {var1}, {layers}layer.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(dicts)
            dicts=[]
            for seed in range(1,11): 
                y_past, y_true, y_pred, forecast_index, y_past_diff_log, y_true_diff_log, y_pred_diff_log= gru(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, var0, var1, optimizer, loss)
                rmse, mae, r2 = evaluate_forecast(y_true, y_pred, var0)
                diff_log_rmse, diff_log_mae, diff_log_r2 = evaluate_forecast(y_true_diff_log, y_pred_diff_log, var0)
                dic={
                     'rmse': rmse,
                     'mae': mae,
                     'r2': r2,
                     'diff_log_rmse': diff_log_rmse,
                     'diff_log_mae': diff_log_mae,
                     'diff_log_r2': diff_log_r2,
                }
                dicts.append(dic)
            header = dicts[0].keys()
            with open(f'results/forecast/csv/gru, {var0}, {var1}, {layers}layer.csv.csv', 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(dicts)




if __name__ == "__main__":
    main()