import subprocess
import sys

hint = "Usage: python main.py <activity> <model_name> \n" \
"   [--seed         <type=int, default=1>] \n" \
"   [--horizon      <type=int, default=12>] \n" \
"   [--lag          <type=int, default=12>] \n" \
"   [--maxlags      <type=int, default=15>] \n" \
"   [--neurons      <type=int, default=200>] \n" \
"   [--batch_size   <type=int, default=16>] \n" \
"   [--epochs       <type=int, default=100>] \n" \
"   [--data_file    <type=str, default='assets/data.json'>] \n" \
"   [--var0         <type=str, default='cpi'>] \n" \
"   [--var1         <type=str, default='ppi'>] \n" \
"   [--ic           <type=str, default='aic'>] \n" \
"   [--optimizer    <type=str, default='adam'>] \n" \
"   [--loss         <type=str, default='mean_squared_error'>]"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(hint)
        print("Example: python run.py forecast gru --var0 'ppi' --var1 'inflation'")
        
        sys.exit(1)

    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    script_arguments = sys.argv[3:]

    command = ["python", "main.py", arg1, arg2, *script_arguments]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        sys.exit(1)

