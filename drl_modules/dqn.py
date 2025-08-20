import os
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from drl_modules.env import TradingEnv
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter.filedialog import askopenfilename
from drl_modules.callbacks import LoggingCallback, EventCallback
from drl_modules.export_model import export_to_onnx, export_to_onnx_dict

remote_url = "https://your-server.com/api/logs"  # Replace with actual server URL
NORM = False


def dqn_run(dir,
            reward_func_idx,
            symbol: str,
            step_amount: int = 0,
            df_path: str | pd.DataFrame = "",
            batch_idx: int = 0,
            batches_amount: int = 0,
            save_model_after_each_batch: bool = False,
            device: str = "cpu",
            agent_policy: str = "MlpPolicy"):
    training_steps = step_amount

    base_env = TradingEnv(reward_func_idx=reward_func_idx,
                          agent_policy=agent_policy,
                          dataset_path=df_path,
                          symbol=symbol,
                          normalize=NORM,
                          discrete_actions=True)
    base_env._get_env_details()
    env = base_env

    parent_dir = os.path.dirname(dir) + "/"
    tmp_path = dir + f"part{batch_idx}/" if batches_amount > 0 else dir

    new_logger = configure(tmp_path + "sb3_log/", ["stdout", "csv", "tensorboard"])

    model = DQN(agent_policy, env, verbose=1, device=device)
    model.set_logger(new_logger)

    log_callback = LoggingCallback(log_dir=tmp_path, dataset_size=base_env.df_size, remote_url=remote_url)
    eval_callback = EventCallback(log_callback)

    model.learn(total_timesteps=training_steps, callback=log_callback)
    model.save(parent_dir + "DQN_model")

    if save_model_after_each_batch:
        model.save(tmp_path + f"batch/DQN_model_batch_{batch_idx}")

    # Export model to ONNX
    if agent_policy == "MultiInputPolicy":
        export_to_onnx_dict(parent_dir + "DQN_model.zip")
    else:
        export_to_onnx(parent_dir + "DQN_model.zip")

    obs, _ = base_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = base_env.step(action)
        if dones:
            base_env._save_trade_env_to_csv(save_directory=tmp_path)
            base_env._save_positions_into_csv(save_directory=tmp_path)
            break

    base_env.render(save_directory=tmp_path)


def dqn_eval(dir: str,
             episodes: int,
             reward_func_idx: int,
             symbol: str,
             model_path: str = "",
             render_modulo: int = 10,
             df_path: str | pd.DataFrame = "",
             device: str = "cpu",
             agent_policy: str = "MlpPolicy",
             eval_only_setting: bool = False):
    env = TradingEnv(reward_func_idx=reward_func_idx,
                     agent_policy=agent_policy,
                     symbol=symbol,
                     dataset_path=df_path,
                     normalize=NORM,
                     discrete_actions=True)

    tmp_path = dir + "evaluation/"
    os.makedirs(tmp_path, exist_ok=True)

    if model_path == "":
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file.",
        )

    model = DQN.load(model_path, env=env, device=device)

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
            "TotalTrades") != 0 else 0

        if e % render_modulo == 0:
            times_mean = np.mean(times)
            times.clear()
            print(f"(DQN) Episode: {e}, Total Reward: {total_rw}, Time duration per ep: {times_mean}")
            if not eval_only_setting:
                env.render(save_directory=tmp_path + f"TradingEnv_EP={e}", figure_name="")

    return_data = pd.DataFrame(return_data)
    return_data.to_csv(dir + "eval_results.csv")
