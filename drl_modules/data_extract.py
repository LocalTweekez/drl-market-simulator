import pandas as pd
import datetime
import tkinter as tk
from tkinter.filedialog import askopenfilename
import drl_modules.create_metadata

def extract_data_windows(symbol: str,
                        date_from: datetime.datetime,
                        date_to: datetime.datetime,
                        batch_size: int):
    import MetaTrader5 as mt5

def extract_data():
    root = tk.Tk()
    root.withdraw()
    csvfile = askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select dataset csv file."
    )
    df = pd.read_csv(csvfile)
    
    # Expected columns order
    expected_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    
    # Check if columns are in the correct order
    if not list(df.columns) == expected_columns:
        output_csv_path = "NEW_CSV.csv"
        convert_csv_format(csvfile, output_csv_path)
        df = pd.read_csv(output_csv_path)
    
    # Sort the DataFrame by the 'time' column
    df = df.sort_values(by='time').reset_index(drop=True)
    
    return df

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
    print(df.head())
