# Nowcasting Economic Variables with Web Traffic Using VAR and Deep Learning
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/euro-kim/nowcast/pulls)

## Overview
This is a source code containg simple code to predict economic variables with web traffic data.

1. **Included Models**
    
    Casual Models: VAR <br />
    Forecasting Models: Linear Model, VAR, LSTM, GRU <br />

2. **Usage**

    By default, the data contains monthly CPI, PPI, employment data from South Korea, ranging from 2010.01 to 2025.03. <br />
    By default, the data contains google trends data for keyword '물가' and inflation in South Korea, ranging from 2010.01 to 2025.03. <br />
    To add , add the data in json format in folder at assets/data.json <br />

## Getting Started 

This guide will help you get up and running with the project.

0. **Prerequisites**

    Make sure you have the following installed on your system:
    
    * **Python 3.12+** ([Download Python](https://www.python.org/downloads/))
    * **pip** (should come with your Python installation)


1.  **Clone the repository:**

    ```bash
    git clone https://github.com/euro-kim/nowcast
    cd nowcast
    ```

2.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    This command will install all the necessary libraries listed in the `requirements.txt` file.
    
4. **Running the Main Script:**

    The primary entry point for this project is the `run.py` script. You can use it to perform different activities with various models.

    The basic syntax is:
    
    ```bash
    python run.py <activity> <model_name> [arguments]
    
    ```
    activity includes 'forecast' and 'casual', and model_name includes  


    ```bash
    python run.py forecast gru --var0 'ppi' --var1 'inflation' --seed 1
    python run.py casual var --var0 'cpi' --var1 'inflation' 
    
    ```


5. **Detail the Available Arguments:**

    A clear table or list is the best way to present the command-line arguments.
            
    ```markdown

    ### Available Acitity
    
    The `run.py` script requires one of the activities:
    
    | Activity      | Description                                                                     |
    |---------------|---------------------------------------------------------------------------------|
    | forecast      | Forecasting (prediction)                                                        |
    | casual        | Casual Inference. Only supports VAR                                             |
                                                                    |

    ### Available Models
    
    The `run.py` script requires one of the following model_names:
    
    | model_name    | Description                                                                     |
    |---------------|---------------------------------------------------------------------------------|
    | arima         | ARIMA model                                                                     |
    | linear        | Simple Linear Regression                                                        |
    | var           | VAR model                                                                       |
    | LSTM          | LSTM model                                                                      |
    | GRU           | GRU model                                                                       |


    
    ### Available Arguments
    
    The `run.py` script accepts the following optional arguments:
    
    | Argument      | Type    | Default               | Description                                                                     |
    |---------------|---------|-----------------------|---------------------------------------------------------------------------------|
    | `--seed`      | `int`   | `1`                   | Random seed for reproducibility.                                                |
    | `--horizon`   | `int`   | `12`                  | The number of time steps to forecast.                                           |
    | `--lag`       | `int`   | `12`                  | The number of lagged observations to use for VAR.                               |
    | `--maxlags`   | `int`   | `15`                  | The maximum number of lags to consider for VAR order selection.                 |
    | `--neurons`   | `int`   | `200`                 | The number of neurons in the deep learning model's layers.                      |
    | `--layerss`   | `int`   | `1`                   | The number of layers for RNN models.                                            |
    | `--batch_size`| `int`   | `16`                  | The batch size for training deep learning models.                               |
    | `--epochs`    | `int`   | `100`                 | The number of training epochs for deep learning.                                |
    | `--data_file` | `str`   | `'assets/data.json'`  | Path to the JSON data file containing your economic and web traffic data.       |
    | `--var0`      | `str`   | `'cpi'`               | The name of the first economic variable.                                        |
    | `--var1`      | `str`   | `'ppi'`               | The name of the second economic variable.                                       |
    | `--ic`        | `str`   | `'aic'`               | The information criterion for VAR order selection (`aic`, `bic`, `hqic`).       |
    | `--optimizer` | `str`   | `'adam'`              | The optimization algorithm for deep learning (`adam`, `sgd`).                   |
    | `--loss`      | `str`   | `'mean_squared_error'`| The loss function for deep learning (`mean_squared_error`, `mae`).              |
    ```
