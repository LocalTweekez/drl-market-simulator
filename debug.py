import pandas as pd
from drl_modules.env import TradingEnv
from drl_modules.agent_render import render_loss_function
from drl_modules.data_extract import extract_data, extract_batched_data
from drl_modules.ppo import ppo_run, ppo_eval
from drl_modules.input_config import get_user_input
from drl_modules.rewards import RewardFunctions
import os
from drl_modules.export_model import export_to_onnx_dict
from stable_baselines3 import PPO

env = TradingEnv(0, "EURUSD", "MlpPolicy", "datasets/eurusd100_real.csv", 10, 10000, 2, 0.01)
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
tmp_path = "debug/"
print(env._get_observation())
