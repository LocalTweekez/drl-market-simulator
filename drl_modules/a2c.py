import os
import time
import tkinter as tk
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from drl_modules.callbacks import EventCallback, LoggingCallback
from drl_modules.env import TradingEnv
from drl_modules.data_extract import extract_data, extract_batched_data
from drl_modules.export_model import export_to_onnx, export_to_onnx_dict


remote_url = "https://your-server.com/api/logs"  # Replace with your actual server URL
NORM = False


def make_env(env_id, reward_idx, agent_policy, df_path, symbol):
    def _init():
        return TradingEnv(
            reward_func_idx=reward_idx,
            agent_policy=agent_policy,
            dataset_path=df_path,
            symbol=symbol,
        )

    return _init


def a2c_run(
    dir,
    reward_func_idx,
    symbol: str,
    step_amount: int = 0,
    df_path: str | pd.DataFrame = "",
    batch_idx: int = 0,
    batches_amount: int = 0,
    save_model_after_each_batch: bool = False,
    vectorized_environments: int = 4,
    device: str = "cpu",
    agent_policy: str = "MultiInputPolicy",
):
    """Train an A2C agent and optionally export to ONNX."""

    training_steps = step_amount

    base_env = TradingEnv(
        reward_func_idx=reward_func_idx,
        agent_policy=agent_policy,
        dataset_path=df_path,
        symbol=symbol,
        normalize=NORM,
    )
    base_env._get_env_details()

    if vectorized_environments > 0:
        vec_env = make_vec_env(
            lambda: make_env(None, reward_func_idx, agent_policy, df_path, symbol)(),
            n_envs=vectorized_environments,
        )
        print("VECTORIZING ENVIRONMENTS (This could cause crashes!)")

    env = base_env if vectorized_environments == 0 else vec_env

    parent_dir = os.path.dirname(dir) + "/"
    tmp_path = dir + f"part{batch_idx}/" if batches_amount > 0 else dir
    print("parent_dir: ", parent_dir)

    new_logger = configure(tmp_path + "sb3_log/", ["stdout", "csv", "tensorboard"])

    model = None
    if batches_amount >= 4:
        try:
            model = A2C.load(parent_dir + "A2C_model", env=env, device=device)
        except FileNotFoundError:
            print("File not found, training a new file instead")
            model = A2C(agent_policy, env, verbose=1, device=device)
    else:
        model = A2C(agent_policy, env, verbose=1, device=device)

    model.set_logger(new_logger)

    log_callback = LoggingCallback(
        log_dir=tmp_path, dataset_size=base_env.df_size, remote_url=remote_url
    )
    eval_callback = EventCallback(log_callback)

    model.learn(total_timesteps=training_steps, callback=log_callback)
    model.save(parent_dir + "A2C_model")

    print(f"Saved model in path {parent_dir+'A2C_model'}")

    # Export to ONNX
    model_zip_path = parent_dir + "A2C_model.zip"
    if agent_policy == "MultiInputPolicy":
        export_to_onnx_dict(model_zip_path, algorithm="A2C")
    else:
        export_to_onnx(model_zip_path, algorithm="A2C")

    if save_model_after_each_batch:
        model.save(tmp_path + f"batch/A2C_model_batch_{batch_idx}")

    obs, _ = base_env.reset()

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = base_env.step(action)
        if dones:
            base_env._save_trade_env_to_csv(save_directory=tmp_path)
            base_env._save_positions_into_csv(save_directory=tmp_path)
            break

    base_env.render(save_directory=tmp_path)


def a2c_eval(
    dir: str,
    episodes: int,
    reward_func_idx: int,
    symbol: str,
    model_path: str = "",
    render_modulo: str = 10,
    df_path: str | pd.DataFrame = "",
    device: str = "cpu",
    agent_policy: str = "MultiInputPolicy",
    eval_only_setting: bool = False,
):
    env = TradingEnv(
        reward_func_idx=reward_func_idx,
        agent_policy=agent_policy,
        symbol=symbol,
        dataset_path=df_path,
        normalize=NORM,
    )

    tmp_path = dir + "evaluation/"
    os.makedirs(tmp_path, exist_ok=True)

    if model_path == "":
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file.",
        )

    model = A2C.load(model_path, env=env, device=device)

    dtype = [
        ("episode", int),
        ("total_reward", float),
        ("total_trades", int),
        ("win_rate", float),
    ]
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
        return_data["win_rate"][e] = (
            info.get("Wins") / info.get("TotalTrades") * 100 if info.get("TotalTrades") != 0 else 0
        )

        if e % render_modulo == 0:
            times_mean = np.mean(times)
            times.clear()
            print(
                f"(A2C) Episode: {e}, Total Reward: {total_rw}, Time duration per ep: {times_mean}"
            )
            if not eval_only_setting:
                env.render(
                    save_directory=tmp_path + f"TradingEnv_EP={e}", figure_name=""
                )

    return_data = pd.DataFrame(return_data)
    return_data.to_csv(dir + "eval_results.csv")


if __name__ == "__main__":
    path = "../results/"
    batch_divider = 1
    start_part = 0

    df = extract_data()
    batches = extract_batched_data(df, batch_divider=batch_divider)

    step_inp = int(input("Enter amount of steps: "))

    last_found_part = -1
    for i in range(batch_divider):
        if os.path.exists(path + f"part{i}"):
            last_found_part = i
        else:
            break

    start_part = last_found_part + 1 if last_found_part != -1 else 0
    print(f"Starting from part {start_part}")

    for i, part in enumerate(batches[start_part:], start=start_part):
        print(f"Training on batch {i + 1}")
        a2c_run(
            dir=path + f"part{i}/",
            reward_func_idx=0,
            symbol="",
            step_amount=step_inp,
            df_path=part,
            batch_idx=i,
            save_model_after_each_batch=True,
            vectorized_environments=6,
            device="cuda",
        )

    eval_part = pd.concat(batches[-4:])
    a2c_eval(
        dir=path + "evaluation/",
        episodes=100,
        reward_func_idx=0,
        symbol="",
        render_modulo=1,
        model_path=path + "A2C_model.zip",
        df_path=eval_part,
        device="cuda",
    )

