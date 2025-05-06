import pandas as pd
import os
import glob

def calculate_average_and_save(directory="."):
    """
    Calculates the average of each column in all CSV files in a directory,
    adds a row with the averages, and saves the modified CSV files.

    Args:
        directory (str, optional): The directory containing the CSV files.
            Defaults to the current directory.
    """
    # Use glob to find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, "results/forecast/csv/*.csv"))

    if not csv_files:
        print(f"No CSV files found in the directory: {directory}")
        return  # Exit if no CSV files are found

    for csv_file in csv_files:
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(csv_file)

            # Check if the DataFrame is empty
            if df.empty:
                print(f"Skipping empty CSV file: {csv_file}")
                continue  # Skip to the next file if the current file is empty

            # Calculate the average of each column
            average_values = df.mean(numeric_only=True)  # numeric_only to avoid errors with non-numeric columns

            # Create a new DataFrame with the average values as a single row
            average_df = pd.DataFrame([average_values])

            # Append the average row to the original DataFrame
            df = pd.concat([df, average_df], ignore_index=True)

            # Save the modified DataFrame back to the same CSV file
            df.to_csv(csv_file, index=False)  # index=False prevents writing the DataFrame index to the CSV
            print(f"Processed and saved: {csv_file}")

        except Exception as e:
            print(f"Error processing file: {csv_file} - {e}")
            # Optionally, you might want to handle the error (e.g., logging, skipping the file)
            # Here, we just print the error and continue to the next file

if __name__ == "__main__":
    # Process CSV files in the current directory
    calculate_average_and_save()
    print("Finished processing CSV files.") # Addded a finishing message
