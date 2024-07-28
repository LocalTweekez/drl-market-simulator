import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from drl_modules.env import TradingEnv
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter.filedialog import askopenfilename
from drl_modules.callbacks import LoggingCallback, EventCallback, plot_total_rewards
from drl_modules.data_extract import extract_data, extract_batched_data

# IGNORE FOR NOW
remote_url = "https://your-server.com/api/logs"  # Replace with your actual server URL

def make_env(env_id, reward_idx, agent_policy, df_path, symbol):
    def _init():
        return TradingEnv(reward_func_idx=reward_idx, agent_policy=agent_policy, dataset_path=df_path, symbol=symbol)
    return _init

def ppo_run(dir, 
            reward_func_idx, 
            symbol: str, 
            step_amount: int = 0, 
            df_path: str | pd.DataFrame = "", 
            batch_idx: int = 0, 
            batches_amount: int = 0,
            save_model_after_each_batch: bool = False, 
            vectorized_environments: int = 4,
            device: str = "cpu",
            agent_policy: str = "MultiInputPolicy"):
    
    training_steps = step_amount

    # Init environment(s)
    base_env = TradingEnv(reward_func_idx=reward_func_idx, 
                          agent_policy=agent_policy,
                          dataset_path=df_path, 
                          symbol=symbol)
    base_env._get_env_details()
    if vectorized_environments > 0:
        vec_env = make_vec_env(lambda: make_env(None, reward_func_idx, agent_policy, df_path, symbol)(), n_envs=vectorized_environments)
        print("VECTORIZING ENVIRONMENTS (This could cause crashes!)")

    env = base_env if vectorized_environments == 0 else vec_env

    # Set directories
    parent_dir = os.path.dirname(dir) + "/"
    tmp_path = dir + f"part{batch_idx}/" if batches_amount > 0 else dir
    print("parent_dir: ", parent_dir)
    
    # Init agent with logging functions
    new_logger = configure(tmp_path+"sb3_log/", ["stdout", "csv", "tensorboard"])
    model = PPO(agent_policy, env, verbose=1, device=device)
    model.set_logger(new_logger)

    if batch_idx > 0 and batches_amount >= 4:
        try:
            model.load(parent_dir+"PPO_model", device=device)
        except FileNotFoundError:
            print("File not found, training a new file instead")

    log_callback = LoggingCallback(log_dir=tmp_path, dataset_size=base_env.df_size, remote_url=remote_url)
    eval_callback = EventCallback(log_callback)

    model.learn(total_timesteps=training_steps, callback=log_callback)
    if batches_amount >= 4:
        model.save(path=parent_dir+"PPO_model")
    else:
        model.save(path=tmp_path+"PPO_model")
    print(f"Saved model in path {tmp_path}")

    if save_model_after_each_batch:
        model.save(tmp_path+f"batch/PPO_model_batch_{batch_idx}")

    obs, _ = base_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = base_env.step(action)
        if dones:
            base_env._save_trade_env_to_csv(save_directory=tmp_path)
            base_env._save_positions_into_csv(save_directory=tmp_path)
            break

    base_env.render(save_directory=tmp_path)


def ppo_eval(dir: str, 
             episodes: int, 
             reward_func_idx: int, 
             symbol: str, 
             model_path: str = "", 
             render_modulo: str = 10, 
             df_path: str | pd.DataFrame = "",
             device: str = "cpu",
             agent_policy: str = "MultiInputPolicy"):
    env = TradingEnv(reward_func_idx=reward_func_idx, 
                     agent_policy=agent_policy,
                     symbol=symbol, 
                     dataset_path=df_path)
    
    tmp_path = dir+"evaluation/"
    os.makedirs(tmp_path, exist_ok=True)

    model = PPO(agent_policy, env, verbose=1)
    
    if model_path == "":
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file."
        )

    try:
        model.load(model_path, device=device)
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
                vectorized_environments=6,
                device="cuda")
                
    eval_part = pd.concat(batches[-4:])
    ppo_eval(dir=res_path+f"evaluation/", 
             episodes=100, 
             reward_func_idx=0,
             render_modulo=1,
             model_path="PPO_model",
             df_path=eval_part,
             device="cuda")
