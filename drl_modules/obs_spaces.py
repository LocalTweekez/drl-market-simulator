import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd
import drl_modules.data_extract as de

# Observation space: open, high, low, close, volume, position type
def get_obs_space_dict(df: pd.DataFrame, box_length: int = 10):

    print("\nCreating DICTIONARY observation space!\n")

    if 'time' in df.columns:
        df = df.drop("time", axis=1)
    else:
        print("Column 'time' not found in DataFrame")

    if 'spread' in df.columns:
        df = df.drop("spread", axis=1)
    else:
        print("Column 'spread' not found in DataFrame")

    if 'real_volume' in df.columns:
        df = df.drop("real_volume", axis=1)
    else:
        print("Column 'real_volume' not found in DataFrame")
    
    std_open = np.std(df["open"])
    std_high = np.std(df["high"])
    std_low = np.std(df["low"])
    std_close = np.std(df["close"])
    std_vol = np.std(df["tick_volume"])

    low_dict = {
        'open': np.array([df["open"].min() - std_open] * box_length, dtype=np.float32),
        'high': np.array([df["high"].min() - std_high] * box_length, dtype=np.float32),
        'low': np.array([df["low"].min() - std_low] * box_length, dtype=np.float32),
        'close': np.array([df["close"].min() - std_close] * box_length, dtype=np.float32),
        'volume': np.array([0] * box_length, dtype=np.float32),
        'position': np.array([0] * box_length, dtype=np.int32)  # Assuming position type is a single value
    }
    
    high_dict = {
        'open': np.array([df["open"].max() + std_open] * box_length, dtype=np.float32),
        'high': np.array([df["high"].max() + std_high] * box_length, dtype=np.float32),
        'low': np.array([df["low"].max() + std_low] * box_length, dtype=np.float32),
        'close': np.array([df["close"].max() + std_close] * box_length, dtype=np.float32),
        'volume': np.array([df["tick_volume"].max() + std_vol] * box_length, dtype=np.float32),
        'position': np.array([2] * box_length, dtype=np.int32)  # Assuming position type is a single value
    }
    
    return spaces.Dict({
        'open': spaces.Box(low=low_dict['open'], high=high_dict['open'], dtype=np.float32),
        'high': spaces.Box(low=low_dict['high'], high=high_dict['high'], dtype=np.float32),
        'low': spaces.Box(low=low_dict['low'], high=high_dict['low'], dtype=np.float32),
        'close': spaces.Box(low=low_dict['close'], high=high_dict['close'], dtype=np.float32),
        'volume': spaces.Box(low=low_dict['volume'], high=high_dict['volume'], dtype=np.float32),
        'position': spaces.Box(low=low_dict['position'], high=high_dict['position'], dtype=np.int32),
    })

# Observation space: open, high, low, close, volume, position type
def get_obs_space_flattened(df: pd.DataFrame, box_length: int = 10):

    print("\nCreating FLATTENED observation space!\n")

    if 'time' in df.columns:
        df = df.drop("time", axis=1)
    else:
        print("Column 'time' not found in DataFrame")

    if 'spread' in df.columns:
        df = df.drop("spread", axis=1)
    else:
        print("Column 'spread' not found in DataFrame")

    if 'real_volume' in df.columns:
        df = df.drop("real_volume", axis=1)
    else:
        print("Column 'real_volume' not found in DataFrame")
    
    std_open = np.std(df["open"])
    std_high = np.std(df["high"])
    std_low = np.std(df["low"])
    std_close = np.std(df["close"])
    std_vol = np.std(df["tick_volume"])

    low_values = np.concatenate([
        [df["open"].min() - std_open] * box_length,
        [df["high"].min() - std_high] * box_length,
        [df["low"].min() - std_low] * box_length,
        [df["close"].min() - std_close] * box_length,
        [0] * box_length,  # Volume
        [0] * box_length  # Position type
    ]).astype(np.float32)
    
    high_values = np.concatenate([
        [df["open"].max() + std_open] * box_length,
        [df["high"].max() + std_high] * box_length,
        [df["low"].max() + std_low] * box_length,
        [df["close"].max() + std_close] * box_length,
        [df["tick_volume"].max() + std_vol] * box_length,  # Volume
        [2] * box_length  # Position type
    ]).astype(np.float32)
    
    return spaces.Box(low=low_values, high=high_values, dtype=np.float32)
