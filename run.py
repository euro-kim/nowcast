import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import subprocess
import sys

hint = "Usage: python run.py <activity> <model_name> \n" \
"   [--seed         <type=int, default=1>] \n" \
"   [--horizon      <type=int, default=12>] \n" \
"   [--lag          <type=int, default=12>] \n" \
"   [--p            <type=int, default=1>] \n" \
"   [--q            <type=int, default=1>] \n" \
"   [--maxlags      <type=int, default=15>] \n" \
"   [--neurons      <type=int, default=200>] \n" \
"   [--layers       <type=int, default=1>] \n" \
"   [--batch_size   <type=int, default=16>] \n" \
"   [--epochs       <type=int, default=100>] \n" \
"   [--data_file    <type=str, default='assets/data.csv'>] \n" \
"   [--vars         <type=str, default='cpi,ppi'>] \n" \
"   [--ic           <type=str, default='aic'>] \n" \
"   [--optimizer    <type=str, default='adam'>] \n" \
"   [--loss         <type=str, default='mean_squared_error'>] \n" \
"   [--compare_models] \n" \
"   [--compare_var_arima_gru] \n" \
"   [--compare_var_lstm_gru]"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(hint)
        print("Example: python run.py forecast gru --vars 'ppi,inflation'")
        sys.exit(1)

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    script_arguments = sys.argv[3:]

    # Use absolute path for main.py to avoid issues
    main_py_path = os.path.join(os.path.dirname(__file__), "main.py")
    command = [sys.executable, main_py_path, arg1, arg2] + script_arguments

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        sys.exit(1)

# Example usage:
vars = ['ppi', 'cpi', 'inflation']
# result = lstm(seed, horizon, lag, neurons, layers, epochs, batch_size, data_file, vars, optimizer, loss)
# result = var(seed, maxlags, horizon, data_file, vars, ic)

