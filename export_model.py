import gymnasium as gym
import onnx
import torch
import numpy as np
import torch as th
from typing import Tuple, Dict
from drl_modules.env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename

class OnnxableSB3Policy(th.nn.Module):
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
        
        observation = th.cat([open, high, low, close, volume, position], dim=1)
        return self.policy._predict(observation, deterministic=False)

if __name__ == "__main__":
    # Example: model = PPO("MlpPolicy", "Pendulum-v1")
    env = TradingEnv(0, "EURUSD", "datasets/EURUSD.csv")
    env.reset()
    model = PPO("MultiInputPolicy", env, device="cuda")

    model = PPO.load("results/16/PPO_model.zip", device="cuda")

    onnx_policy = OnnxableSB3Policy(model.policy)

    # Extract observation space sizes
    observation_space = env.observation_space
    dummy_input = {
        'open': th.randn(1, 10).to("cuda"),
        'high': th.randn(1, 10).to("cuda"),
        'low': th.randn(1, 10).to("cuda"),
        'close': th.randn(1, 10).to("cuda"),
        'volume': th.randn(1, 10).to("cuda"),
        'position': th.randn(1, 10).to("cuda")
    }

    th.onnx.export(
        onnx_policy,
        (dummy_input['open'], dummy_input['high'], dummy_input['low'], dummy_input['close'], dummy_input['volume'], dummy_input['position']),
        "my_ppo_model.onnx",
        opset_version=17,
        input_names=["open", "high", "low", "close", "volume", "position"],
        output_names=["output"],
    )

    ##### Load and test with onnx

    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_path = "my_ppo_model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    observation = {
        'open': np.zeros((1, 10)).astype(np.float32),
        'high': np.zeros((1, 10)).astype(np.float32),
        'low': np.zeros((1, 10)).astype(np.float32),
        'close': np.zeros((1, 10)).astype(np.float32),
        'volume': np.zeros((1, 10)).astype(np.float32),
        'position': np.zeros((1, 10)).astype(np.int32)
    }
    ort_sess = ort.InferenceSession(onnx_path)
    inputs = {
        "open": observation['open'],
        "high": observation['high'],
        "low": observation['low'],
        "close": observation['close'],
        "volume": observation['volume'],
        "position": observation['position']
    }
    outputs = ort_sess.run(None, inputs)

    print(outputs)

    # Check that the predictions are the same
    with th.no_grad():
        pytorch_inputs = {
            'open': th.as_tensor(observation['open']).to("cuda"),
            'high': th.as_tensor(observation['high']).to("cuda"),
            'low': th.as_tensor(observation['low']).to("cuda"),
            'close': th.as_tensor(observation['close']).to("cuda"),
            'volume': th.as_tensor(observation['volume']).to("cuda"),
            'position': th.as_tensor(observation['position']).to("cuda")
        }
        pytorch_inputs_combined = th.cat([pytorch_inputs['open'], pytorch_inputs['high'], pytorch_inputs['low'], pytorch_inputs['close'], pytorch_inputs['volume'], pytorch_inputs['position']], dim=1)
        pytorch_outputs = model.policy(pytorch_inputs_combined, deterministic=False)
        print(pytorch_outputs)
