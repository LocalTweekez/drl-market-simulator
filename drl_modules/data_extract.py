import pandas as pd
import datetime
import tkinter as tk
from tkinter.filedialog import askopenfilename
import drl_modules.create_metadata
import numpy as np

def extract_data_windows(symbol: str,
                        date_from: datetime.datetime,
                        date_to: datetime.datetime,
                        batch_size: int):
    import MetaTrader5 as mt5

def get_csv_path_pandas():
    root = tk.Tk()
    root.withdraw()
    csvfile = askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select dataset csv file."
    )
    return csvfile

def extract_data(csv_path=""):
    csvfile = get_csv_path_pandas() if csv_path == "" else csv_path

    df = pd.read_csv(csvfile)
    
    # Columns to check for
    required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    
    # Check if required columns exist
    if all(column in df.columns for column in required_columns):
        # Extract only the required columns
        df = df[required_columns]
    else:
        # Call convert_csv_format if the required columns are not found
        output_csv_path = "NEW_CSV.csv"
        convert_csv_format(csvfile, output_csv_path)
        df = pd.read_csv(output_csv_path)
    
    # Sort the DataFrame by the 'time' column
    df = df.sort_values(by='time').reset_index(drop=True)
    
    return df

def extract_batched_data(df, batch_divider=10, csv_path=""):
    if batch_divider < 4:
        print("Batch divider must be at least 4 (the last two are for evaluation).")
        quit()
        
    num_rows = len(df)
    batch_size = int(np.ceil(num_rows / batch_divider))
    indices = df.index
    batches = [df.loc[indices[i * batch_size: min((i + 1) * batch_size, num_rows)]] for i in
                    range(batch_divider)]
    
    return batches

def convert_csv_format(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    
    # Combine DATE and TIME into a single datetime column
    df['time'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    
    # Rename columns
    df.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close', 'TICKVOL': 'tick_volume'}, inplace=True)
    
    # Drop the original DATE and TIME columns
    df.drop(columns=['DATE', 'TIME'], inplace=True)
    
    # Add spread and real_volume columns with default values 0
    df['spread'] = 0
    df['real_volume'] = 0
    
    # Reorder columns
    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
    
    # Sort the DataFrame by the 'time' column
    df = df.sort_values(by='time').reset_index(drop=True)
    
    # Save the output CSV file
    df.to_csv(output_csv_path, index=False)

# Example usage in your module:
if __name__ == "__main__":
    df = extract_data()
    batches = extract_batched_data(df)
    start_part = 0

    for i, part in enumerate(batches[start_part:], start=start_part):
        print(part.head())
