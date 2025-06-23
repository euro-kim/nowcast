# Nowcasting Economic Variables with Web Traffic Using VAR and Deep Learning
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/euro-kim/nowcast/pulls)

## Overview
This repository provides code to predict economic variables using web traffic data.

**Included Models:**
- Causal Models: VAR  
- Forecasting Models: ARIMA, Linear Model, VAR, LSTM, GRU

## Getting Started

### 0. Prerequisites

- **Python 3.12+** ([Download Python](https://www.python.org/downloads/))
- **pip** (comes with Python)

### 1. Clone the repository

```bash
git clone https://github.com/euro-kim/nowcast
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

- By default, the data contains monthly CPI, PPI, and employment data from South Korea (2010.01–2025.03).
- Google Trends data for the keyword '물가' and inflation in South Korea is also included.
- To use your own data, add it in CSV format to `assets/data.csv`.

### 4. Running the Main Script

The main entry point is `run.py`. You can use it to perform different activities with various models.

**Basic syntax:**
```bash
python run.py <activity> <model_name> [arguments]
```

**Examples:**
```bash
python run.py forecast gru --vars 'ppi,inflation' --seed 1
python run.py casual var --vars 'cpi,inflation'
```

**Model Comparison Examples:**
```bash
python run.py compare var --vars 'cpi,ppi'
python run.py compare_var_lstm_gru var --vars 'cpi,ppi'
python run.py compare_var_arima_gru var --vars 'cpi,ppi'
# Or using flags:
python run.py forecast gru --vars 'cpi,ppi' --compare_models
python run.py forecast gru --vars 'cpi,ppi' --compare_var_lstm_gru
python run.py forecast gru --vars 'cpi,ppi' --compare_var_arima_gru
```

---

## Command-Line Arguments

### Activities

| Activity                  | Description                                              |
|---------------------------|----------------------------------------------------------|
| `forecast`                | Forecasting (prediction)                                 |
| `casual`                  | (removed)                                               |
| `compare`                 | Compare ARIMA, AR, MA, VAR predictions                  |
| `compare_var_lstm_gru`    | Compare VAR, LSTM, GRU predictions                      |
| `compare_var_arima_gru`   | Compare VAR, ARIMA, GRU predictions                     |

### Models

| model_name    | Description                  |
|---------------|-----------------------------|
| arima         | ARIMA model                 |
| linear        | Simple Linear Regression    |
| var           | VAR model                   |
| lstm          | LSTM model                  |
| gru           | GRU model                   |

### Arguments

| Argument                  | Type    | Default                | Description                                                        |
|---------------------------|---------|------------------------|--------------------------------------------------------------------|
| `--seed`                  | int     | 1                      | Random seed for reproducibility                                    |
| `--horizon`               | int     | 12                     | Number of time steps to forecast                                   |
| `--lag`                   | int     | 12                     | Number of lagged observations (for VAR, LSTM, GRU)                 |
| `--p`                     | int     | -1                     | AR order (AR, ARIMA, GARCH)                                       |
| `--d`                     | int     | -1                     | Differencing order (ARIMA)                                         |
| `--q`                     | int     | -1                     | MA order (MA, ARIMA, GARCH)                                       |
| `--maxlags`               | int     | 15                     | Maximum lags for VAR                                               |
| `--neurons`               | int     | 200                    | Number of neurons in RNN layers                                    |
| `--layers`                | int     | 1                      | Number of layers for RNN models                                    |
| `--batch_size`            | int     | 16                     | Batch size for RNN training                                        |
| `--epochs`                | int     | 100                    | Number of training epochs for RNN                                  |
| `--data_file`             | str     | 'assets/data.csv'      | Path to the CSV data file                                          |
| `--vars`                  | str     | 'cpi,ppi'              | Comma-separated list of variables                                  |
| `--ic`                    | str     | 'aic'                  | Information criterion for VAR (`aic`, `bic`, etc.)                 |
| `--optimizer`             | str     | 'adam'                 | Optimizer for RNN models                                           |
| `--loss`                  | str     | 'mean_squared_error'   | Loss function for RNN models                                       |
| `--compare_models`        | flag    |                        | Compare ARIMA, AR, MA, VAR models in a single plot                 |
| `--compare_var_lstm_gru`  | flag    |                        | Compare VAR, LSTM, GRU models in a single plot                     |
| `--compare_var_arima_gru` | flag    |                        | Compare VAR, ARIMA, GRU models in a single plot                    |

---

## Notes

- For model comparison, you can use either the `activity` argument (`compare`, `compare_var_lstm_gru`, `compare_var_arima_gru`) or the corresponding flag (`--compare_models`, `--compare_var_lstm_gru`, `--compare_var_arima_gru`).
- The `--vars` argument should be a comma-separated string of variable names present in your data file.
- All plots and results are saved in the `results/` directory.

---
