import os
import gymnasium as gym
import torch as th
from drl_modules.env import TradingEnv
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import BasePolicy
import onnx
import onnxruntime as ort
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
from typing import Tuple
from drl_modules.data_extract import extract_data
import yaml
import time
import pandas as pd

ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
}

class OnnxableSB3Policy(th.nn.Module):
    def __init__(self, policy: BasePolicy, action_low: np.ndarray, action_high: np.ndarray):
        super().__init__()
        self.policy = policy
        self.action_low = th.tensor(action_low, dtype=th.float32)
        self.action_high = th.tensor(action_high, dtype=th.float32)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        device = next(self.policy.parameters()).device
        observation = observation.to(device=device)
        actions, _, _ = self.policy(observation, deterministic=True)
        clamped_actions = th.clamp(actions, self.action_low, self.action_high)
        return clamped_actions

def export_to_onnx(_model_path: str = "", algorithm: str = "PPO"):
    if not _model_path:
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file."
        )
    else:
        model_path = _model_path

    data = None
    method = os.path.dirname(model_path) + "/"
    with open(method + "configuration.yaml", "r") as f:
        data = yaml.safe_load(f)

    env = TradingEnv(reward_func_idx=data["Reward"],
                     symbol=data["Symbol"],
                     agent_policy=data["Policy"],
                     dataset_path=data["Dataset"])

    model_directory = os.path.dirname(model_path) + "/"
    model_cls = ALGOS.get(algorithm.upper(), PPO)
    model = model_cls.load(model_path, env=env, device=data["Device"])

    # Get the action space boundaries
    action_low = model.action_space.low
    action_high = model.action_space.high

    onnx_policy = OnnxableSB3Policy(model.policy, action_low, action_high)
    onnx_path = model_directory + "ONNX_model.onnx"

    observation_size = model.observation_space.shape

    dummy_input = th.randn(1, *observation_size)
    dummy_input = env.reset()
    dummy_input = th.tensor(dummy_input[0], dtype=th.float32).unsqueeze(0)  # Convert to torch tensor and add batch dimension

    th.onnx.export(
        onnx_policy,
        dummy_input,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=['actions'],
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported to {onnx_path}")

    # Validate the exported model
    validate_onnx_model(onnx_path, env, model)

def validate_pytorch_model(model, env):
    obs, info = env.reset()
    print("Initial PyTorch Observation:", obs)
    done = False
    while not done:
        obs_tensor = th.tensor([obs], dtype=th.float32).to(model.device)
        with th.no_grad():
            actions, _ = model.policy(obs_tensor, deterministic=False)
        print("PyTorch Actions:", actions.cpu().numpy())
        obs, rewards, done, truncated, info = env.step(actions.cpu().numpy()[0])
        print("Next PyTorch Observation:", obs)

def format_observation(observation):
    formatted_values = [
        f'{x:.4f}' if x < 1e5 else f'{x:.0f}'
        for x in observation
    ]
    return ', '.join(formatted_values)

def validate_onnx_model(onnx_path, env, model):
    ort_sess = ort.InferenceSession(onnx_path)
    actions_df = pd.DataFrame(columns=["direction", "risk"])

    obs_base, info = env.reset()
    obs = obs_base.astype(np.float32)
    obs = np.expand_dims(obs, axis=0)
    done = False
    print(obs)

    with open("misc/output_2.csv", "r") as file:
        while not done:
            obs_from_mt5 = file.readline().strip().split(',')
            try:
                floats = [float(value) for value in obs_from_mt5]
            except ValueError:
                floats = obs[0]
            
            actions = ort_sess.run(['actions'], {"input": [floats]})[0]
            model_act, _ = model.predict(obs_base)
            obs_base, rewards, done, truncated, info = env.step(actions[0])
            obs = obs_base.astype(np.float32)
            obs = np.expand_dims(obs, axis=0)
            formatted_obs = format_observation(obs_base)
            

            #print(floats)
            actions_df.loc[-1] = actions[0]
            actions_df.index += 1
            actions_df = actions_df.sort_index()
            #print(info)
            #print("\n")
            
            
            """
            print("Time: ", info["Time"])
            print("ONNX Actions:", actions)
            print("Next ONNX Observation:", format_observation(obs_base))
            print("\n")
            """

    print(actions_df.head(10))
    print(env.info)
    actions_df.to_csv("misc/outputs_actions.csv")

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################



class OnnxableSB3PolicyDict(th.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, open, high, low, close, volume, position):
        device = next(self.policy.parameters()).device
        
        # Move inputs to the same device as the policy
        open = open.to(device)
        high = high.to(device)
        low = low.to(device)
        close = close.to(device)
        volume = volume.to(device)
        position = position.to(device)

        # Create the observation dictionary
        observation = {
            'open': open,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'position': position
        }
        # Get the action distribution
        distribution = self.policy.get_distribution(observation)
        # Sample action from the distribution
        actions = distribution.get_actions(deterministic=False)
        return actions


def export_to_onnx_dict(_model_path: str = "", algorithm: str = "PPO"):
    # Initialize environment and model

    if not _model_path:
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file.",
        )
    else:
        model_path = _model_path

    data = None
    method = os.path.dirname(model_path) + "/"
    with open(method+"configuration.yaml", "r") as f:
        data = yaml.safe_load(f)
    env = TradingEnv(reward_func_idx=data["Reward"],
                     symbol=data["Symbol"],
                     agent_policy=data["Policy"],
                     dataset_path=data["Dataset"],
                     batch_size=data["Batches"])

    model_directory = os.path.dirname(model_path) + "/"
    model_cls = ALGOS.get(algorithm.upper(), PPO)
    model = model_cls.load(model_path, env=env, device=data.get("Device", "cpu"))

    onnx_policy = OnnxableSB3PolicyDict(model.policy)
    onnx_path = model_directory+"ONNX_model.onnx"

    # Extract observation space sizes
    observation_space = env.observation_space
    dummy_input = {
        'open': th.randn(1, *observation_space['open'].shape).to("cuda"),
        'high': th.randn(1, *observation_space['high'].shape).to("cuda"),
        'low': th.randn(1, *observation_space['low'].shape).to("cuda"),
        'close': th.randn(1, *observation_space['close'].shape).to("cuda"),
        'volume': th.randn(1, *observation_space['volume'].shape).to("cuda"),
        'position': th.randn(1, *observation_space['position'].shape).to("cuda")
    }

    th.onnx.export(
        onnx_policy,
        (dummy_input['open'], dummy_input['high'], dummy_input['low'], dummy_input['close'], dummy_input['volume'], dummy_input['position']),
        onnx_path,
        opset_version=17,
        input_names=["open", "high", "low", "close", "volume", "position"],
        output_names=['output'],
    )

    ##### Load and test with onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    observation = {
        'open': np.zeros(observation_space['open'].shape).astype(np.float32),
        'high': np.zeros(observation_space['high'].shape).astype(np.float32),
        'low': np.zeros(observation_space['low'].shape).astype(np.float32),
        'close': np.zeros(observation_space['close'].shape).astype(np.float32),
        'volume': np.zeros(observation_space['volume'].shape).astype(np.float32),
        'position': np.zeros(observation_space['position'].shape).astype(np.float32)
    }
    ort_sess = ort.InferenceSession(onnx_path)
    inputs = {
        "open": observation['open'][np.newaxis, ...],
        "high": observation['high'][np.newaxis, ...],
        "low": observation['low'][np.newaxis, ...],
        "close": observation['close'][np.newaxis, ...],
        "volume": observation['volume'][np.newaxis, ...],
        "position": observation['position'][np.newaxis, ...]
    }
    outputs = ort_sess.run(None, inputs)

    print("ONNX Outputs:", outputs)

    # Check that the predictions are the same
    with th.no_grad():
        pytorch_inputs = {
            'open': th.as_tensor(inputs['open']).to("cuda"),
            'high': th.as_tensor(inputs['high']).to("cuda"),
            'low': th.as_tensor(inputs['low']).to("cuda"),
            'close': th.as_tensor(inputs['close']).to("cuda"),
            'volume': th.as_tensor(inputs['volume']).to("cuda"),
            'position': th.as_tensor(inputs['position']).to("cuda")
        }
        pytorch_outputs = model.policy._predict(pytorch_inputs, deterministic=False)
        print("PyTorch Outputs:", pytorch_outputs)


if __name__ == "__main__":
    export_to_onnx()