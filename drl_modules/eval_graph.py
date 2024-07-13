import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tkinter import Tk
from tkinter.filedialog import askdirectory


def select_directory():
    """Open a dialog to select the directory containing the CSV files."""
    root = Tk()
    root.withdraw()  # Hide the root window
    selected_dir = askdirectory(title="Select Directory with CSV Files")
    root.destroy()
    return selected_dir


def plot_entire_simulation():
    # Select directory using file dialog
    font_scaler = 1.4
    csv_dir = select_directory()
    if not csv_dir:
        print("No directory selected. Exiting.")
        return

    # Collect all CSV file paths containing "TradingEnv" in the filename and excluding those with "SIMRES"
    csv_files = [file for file in glob.glob(os.path.join(csv_dir, '*.csv'))
                 if "TradingEnv" in os.path.basename(file) and "SIMRES" not in os.path.basename(file)]

    # List to store dataframes with their final account balances
    data_with_balances = []

    # Read each CSV file and store the dataframe along with its final account balance
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Check if 'accBalance' column exists
        if 'eAccBalance' not in df.columns:
            print(f"Column 'accBalance' not found in {csv_file}. Skipping this file.")
            continue

        final_balance = df['eAccBalance'].iloc[-1]
        data_with_balances.append((df, final_balance))

    # Sort the list based on the final account balance
    data_with_balances.sort(key=lambda x: x[1])

    # Prepare the plot
    fig, ax = plt.subplots(figsize=(15, 10))

    # Define the range of opacities
    min_opacity = 0.0
    max_opacity = 1.0
    num_files = len(data_with_balances)

    # Calculate the opacities for each file
    opacities = [min_opacity + (max_opacity - min_opacity) * (i / (num_files - 1)) for i in range(num_files)]

    # Plot each series
    for (df, final_balance), opacity in zip(data_with_balances, opacities):
        ax.plot(pd.to_datetime(df['time']), df['eAccBalance'], alpha=opacity)

    # Set labels and title
    ax.set_xlabel('Time', fontsize=14*font_scaler)
    ax.set_ylabel('Account Balance', fontsize=14*font_scaler)
    ax.set_title('Account Balance Over Time - LSTM', fontsize=16*font_scaler)

    # Limit the x-axis to a maximum of 10 timestamps
    x_ticks = pd.to_datetime(df['time']).iloc[::max(len(df) // 10, 1)].tolist()
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x.strftime('%Y-%m-%d') for x in x_ticks], rotation=45, fontsize=10*font_scaler)
    plt.yticks(fontsize=10*font_scaler)

    # Remove legends
    ax.legend().remove()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(csv_dir+'\\performance.png')
    plt.show()


if __name__ == "__main__":
    plot_entire_simulation()
