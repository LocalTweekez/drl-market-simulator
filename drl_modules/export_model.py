import os
import gymnasium as gym
import torch as th
from drl_modules.env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import onnx
import onnxruntime as ort
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename

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

def export_to_onnx(_model_path=""):
    # Initialize environment and model
    env = TradingEnv(0, "EURUSD", "datasets/EURUSD.csv")

    if not _model_path:
        root = tk.Tk()
        root.withdraw()
        model_path = askopenfilename(
            filetypes=[("ZIP files", "*.zip")],
            title="Select model zip file."
        )
    else:
        model_path = _model_path

    model_directory = os.path.dirname(model_path) + "/"
    model = PPO("MultiInputPolicy", env, device="cuda")
    model = PPO.load(model_path, device="cuda")

    onnx_policy = OnnxableSB3Policy(model.policy)
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