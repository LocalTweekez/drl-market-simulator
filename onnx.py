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

    def forward(self, observation):
        print(observation)
        return observation["a"]
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        return self.policy._predict(observation, deterministic=True)

if __name__ == "__main__":
    # Example: model = PPO("MlpPolicy", "Pendulum-v1")
    env = TradingEnv(0, "EURUSD", "datasets/EURUSD.csv")
    env.reset()
    model = PPO("MultiInputPolicy", env)

    model = PPO.load("results/16/PPO_model.zip", device="cuda")

    onnx_policy = OnnxableSB3Policy(model.policy)

    observation_size = env.observation_space.shape
    print(f"\n\n\n{observation_size}")
    dummy_input = th.randn(1, *observation_size)
    th.onnx.export(
        onnx_policy,
        dummy_input,
        "my_ppo_model.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
    )

    ##### Load and test with onnx

    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_path = "my_ppo_model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    observation = np.zeros((1, *observation_size)).astype(np.float32)
    ort_sess = ort.InferenceSession(onnx_path)
    actions, values, log_prob = ort_sess.run(None, {"input": observation})

    print(actions, values, log_prob)

    # Check that the predictions are the same
    with th.no_grad():
        print(model.policy(th.as_tensor(observation), deterministic=True))