import os
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from drl_modules.env import TradingEnv
from stable_baselines3.common.logger import configure
from drl_modules.agent_render import render_loss_function
from drl_modules.data_extract import extract_data, extract_batched_data
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np
import time

def make_env(env_id, reward_idx, df_path, symbol):
    def _init():
        return TradingEnv(reward_func_idx=reward_idx, dataset_path=df_path, symbol=symbol)
    return _init

def ppo_run(dir, reward_func_idx, symbol: str, step_amount=0, df_path: str | pd.DataFrame = "", batch_idx=0, save_model_after_each_batch=False, vectorized_environments=4):
    if step_amount == 0:
        training_steps = int(input("Enter the amount of training steps: "))
    else:
        training_steps = step_amount

    base_env = TradingEnv(reward_func_idx=reward_func_idx, 
                          dataset_path=df_path, symbol=symbol)

    if vectorized_environments > 0:
        # Here we use a lambda to pass the function with the parameters
        vec_env = make_vec_env(lambda: make_env(None, reward_func_idx, df_path, symbol)(), n_envs=vectorized_environments)
        print("VECTORIZING ENVIRONMENTS (This could cause crashes!)")

    env = base_env if vectorized_environments == 0 else vec_env

    tmp_path = dir
    new_logger = configure(tmp_path+"sb3_log/", ["stdout", "csv", "tensorboard"])
    model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")

    model.set_logger(new_logger)

    if batch_idx > 0:
        try:
            model.load(tmp_path+"../PPO_model", device="cpu")
        except FileNotFoundError:
            print("File not found, training a new file instead")

    model.learn(training_steps)
    model.save(tmp_path+"../PPO_model")
    if save_model_after_each_batch:
        model.save(tmp_path+f"PPO_model_batch_{batch_idx}")

    obs, _ = base_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = base_env.step(action)
        if dones:
            base_env._save_trade_env_to_csv(save_directory=tmp_path)
            base_env._save_positions_into_csv(save_directory=tmp_path)
            break

    base_env.render(save_directory=tmp_path)

def ppo_eval(dir, episodes, reward_func_idx, symbol, model_path="", render_modulo=10, df_path: str | pd.DataFrame = ""):
    env = TradingEnv(reward_func_idx=reward_func_idx, symbol=symbol, dataset_path=df_path)
    
    tmp_path = dir+"evaluation/"
    os.makedirs(tmp_path, exist_ok=True)

    model = PPO("MultiInputPolicy", env, verbose=1)
    
    if model_path == "":
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file."
        )

    try:
        model.load(model_path, device="cpu")
    except FileNotFoundError:
        print("File not found, please select a proper model for evaluation.")
        quit()

    dtype = [('episode', int),
                ('total_reward', float),
                ('total_trades', int),
                ('win_rate', float)]
    return_data = np.zeros(episodes, dtype=dtype)
    total_rewards = []
    
    for e in range(episodes):
        obs, _ = env.reset()
        total_rw = 0
        times = []
        start_time = time.time()
        
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action=action)
            total_rw += rewards
            if dones:
                break

        total_rewards.append(total_rw)
        end_time = time.time()
        times.append(end_time - start_time)

        return_data["episode"][e] = e
        return_data["total_reward"][e] = total_rw
        return_data["total_trades"][e] = info.get("TotalTrades")
        return_data["win_rate"][e] = info.get("Wins") / info.get("TotalTrades") * 100 if info.get(
            "TotalTrades") != 0 else 0  # in percent

        if e % render_modulo == 0:
            times_mean = np.mean(times)
            times.clear()
            print(f"(PPO) Episode: {e}, Total Reward: {total_rw}, Time duration per ep: {times_mean}")
            env.render(save_directory=tmp_path + f"TradingEnv_EP={e}", figure_name="")

    return_data = pd.DataFrame(return_data)
    return_data.to_csv(dir+"eval_results.csv")

def get_path_from_input():
    folder = input("Enter name of the folder to save the results in: ")
    res_path = path+folder+"/"
    os.makedirs(res_path, exist_ok=True)
    return res_path

if __name__ == "__main__":
    path = "../results/"
    batch_divider = 1
    start_part = 0

    df = extract_data()
    batches = extract_batched_data(df, batch_divider=batch_divider)
    res_path = get_path_from_input()
    
    step_inp = int(input("Enter amount of steps: "))
    
    last_found_part = -1
    for i in range(batch_divider):
        if os.path.exists(res_path+f"part{i}"):
            last_found_part = i
        else:
            break

    start_part = last_found_part + 1 if last_found_part != -1 else 0
    print(f"Starting from part {start_part}")

    for i, part in enumerate(batches[start_part:], start=start_part):
        print(f"Training on batch {i + 1}")
        ppo_run(dir=res_path+f"part{i}/", 
                reward_func_idx=0, 
                step_amount=step_inp, 
                df_path=part,
                batch_idx=i,
                save_model_after_each_batch=True,
                vectorized_environments=6)
                
    eval_part = pd.concat(batches[-4:])
    ppo_eval(dir=res_path+f"evaluation/", 
             episodes=100, 
             reward_func_idx=0,
             render_modulo=1,
             df_path=eval_part)
