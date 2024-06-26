import pandas as pd
import datetime
import tkinter as tk
from tkinter.filedialog import askopenfilename
import drl_modules.create_metadata

def extract_data_windows(symbol: str,
                        date_from: datetime,
                        date_to: datetime,
                        batch_size: int):
    import MetaTrader5 as mt5

def extract_data():
    root = tk.Tk()
    root.withdraw()
    csvfile = askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select dataset csv file."
    )
    return pd.read_csv(csvfile)
