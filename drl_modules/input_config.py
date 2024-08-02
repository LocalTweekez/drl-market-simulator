import pandas as pd
from drl_modules.rewards import RewardFunctions
import os
import tkinter as tk
from tkinter.filedialog import askdirectory
import yaml

def get_path_from_input(path):
    folder = input("Enter name of the folder to save the results in: ")
    res_path = os.path.join(path, folder)
    os.makedirs(res_path, exist_ok=True)
    return res_path

def get_user_sim_method():
    choose_sim = input("Perform complete run with training? [y/n]")
    if choose_sim == "n":
        root = tk.Tk()
        root.withdraw()
        folder_path = askdirectory(
            title="Select directory to save results in."
        )
        return folder_path+"/"
    return ""

def get_user_input():
    batches = int(input("Enter amount of batches (zero for single, minimum 4): "))

    print("\nDatasets:")
    dfs = []
    for i, f in enumerate(os.listdir("datasets/")):
        print(f"\t{i}. {f}")
        dfs.append(f)
    df_idx = int(input("\nEnter dataset index: "))
    df = dfs[df_idx]

    file_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(file_dir)
    df_dir = os.path.join(parent_dir, "datasets")
    df = os.path.join(df_dir, df)

    print("\nReward functions:")
    rewards = RewardFunctions().function_names
    for i, r in enumerate(rewards):
        print(f"\t{i}. {r}")
    rw_idx = int(input("\nEnter reward function index: "))

    symbol = input("Enter currency pair symbol in 6 capital letters: ")

    step_inp = int(input("Enter amount of training steps: "))

    vec_env = int(input("Enter amount of vectorized environments (zero for base): "))

    device_idx = int(input("Enter device (0 - cuda, 1 - cpu): "))
    device = "cuda" if device_idx == 0 else "cpu" if device_idx == 1 else "auto"

    res_path = input("Set results folder manually? [y/n]: ")

    if res_path == "y":
        root = tk.Tk()
        root.withdraw()
        folder_path = askdirectory(
            title="Select directory to save results in."
        )
    else:
        name_init = f"A0_{batches}{rw_idx}{vec_env}_{step_inp}_{symbol}"
        res_path = os.path.join(parent_dir, "results", "autosave")
        os.makedirs(res_path, exist_ok=True)

        existing_folders = os.listdir(res_path)
        if existing_folders:
            # Find the maximum index
            max_index = -1
            for folder in existing_folders:
                if folder.startswith('A'):
                    try:
                        index = int(folder[1:].split('_')[0])
                        if index > max_index:
                            max_index = index
                    except ValueError:
                        pass
            next_index = max_index + 1
            name_init = f"A{next_index}_{batches}{rw_idx}{vec_env}_{step_inp}_{symbol}"
        folder_path = os.path.join(res_path, name_init)
        os.makedirs(folder_path, exist_ok=True)

    folder_path += "/"
    print("Saving results in folder:", folder_path)

    policies = ["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
    for i, p in enumerate(policies):
        print(f"\t\t{i}. {p}")
    p_idx = int(input("\nEnter Policy index: "))

    # Create the data dictionary
    data = {
        "Symbol": symbol,
        "Batches": batches,
        "Reward": rw_idx,
        "Dataset": df,
        "Steps": step_inp,
        "VectEnvs": vec_env,
        "Device": device,
        "Folder": folder_path,
        "Policy": policies[p_idx]
    }

    # Write the data to a YAML file
    yaml_path = os.path.join(folder_path, "configuration.yaml")
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file)

    print(f"Configuration saved to {yaml_path}")

    return data

if __name__ == "__main__":
    get_user_input()
