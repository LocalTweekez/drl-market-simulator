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
import onnxruntime as ort


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
    env = TradingEnv(reward_func_idx=0,
                        dataset_path='datasets/EURUSD.csv',
                        batch_size=10)

    root = tk.Tk()
    root.withdraw()
    path = askopenfilename(
        filetypes=[("ZIP files", "*.zip")],
        title="Select model zip file."
    )

    model_zip_path = path
    output_onnx_path = path.replace(".zip", ".onnx")

    obs = env.reset()
    # Example: model = PPO("MlpPolicy", "Pendulum-v1")
    model = PPO.load(model_zip_path, device="cpu")

    onnx_policy = OnnxableSB3Policy(model.policy)

    observation_size = model.observation_space.shape
    # Add batch dimension
    dummy_input = {
        # "a": np.array(obs["a"])[np.newaxis, ...],
        "a": np.array(obs["a"]),
        # "b": np.array(obs["b"])[np.newaxis, ...],
    }
    dummy_input_tensor = {
        "a": th.as_tensor(dummy_input["a"]),
        # "b": th.as_tensor(dummy_input["b"]),
    }

    print(model.predict(dummy_input, deterministic=False))


    th.onnx.export(
        onnx_policy,
        args=(dummy_input_tensor, {}),
        f="my_ppo_model.onnx",
        opset_version=17,
        input_names=["input"],
    )

    ##### Load and test with onnx


    onnx_path = "my_ppo_model.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    observation = dummy_input.copy()
    ort_sess = ort.InferenceSession(onnx_path)

    # print(ort_sess.get_inputs()[0].name)
    # print(ort_sess.get_inputs())

    output = ort_sess.run(None, {"input": observation})

    print(output)

    # Check that the predictions are the same
    # with th.no_grad():
    #     print(model.policy(th.as_tensor(observation), deterministic=False))


"""
import torch as th
from typing import Tuple
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

import onnx
import onnxruntime as ort
import numpy as np


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

env = TradingEnv(reward_func_idx=0,
                    dataset_path='datasets/EURUSD.csv',
                    batch_size=10)

root = tk.Tk()
root.withdraw()
path = askopenfilename(
    filetypes=[("ZIP files", "*.zip")],
    title="Select model zip file."
)

model_zip_path = path
output_onnx_path = path.replace(".zip", ".onnx")

convert_sb3_model_to_onnx(env, model_zip_path, output_onnx_path)

obs, _ = env.reset()
# Example: model = PPO("MlpPolicy", "Pendulum-v1")
model = PPO.load(model_zip_path, device="cpu")

onnx_policy = OnnxableSB3Policy(model.policy)

observation_size = model.observation_space.shape
# Add batch dimension
dummy_input = {
    # "a": np.array(obs["a"])[np.newaxis, ...],
    "a": np.array(obs["a"]),
    # "b": np.array(obs["b"])[np.newaxis, ...],
}
dummy_input_tensor = {
    "a": th.as_tensor(dummy_input["a"]),
    # "b": th.as_tensor(dummy_input["b"]),
}

print(model.predict(dummy_input, deterministic=False))


th.onnx.export(
    onnx_policy,
    args=(dummy_input_tensor, {}),
    f="my_ppo_model.onnx",
    opset_version=17,
    input_names=["input"],
)

##### Load and test with onnx


onnx_path = "my_ppo_model.onnx"
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

observation = dummy_input.copy()
ort_sess = ort.InferenceSession(onnx_path)

# print(ort_sess.get_inputs()[0].name)
# print(ort_sess.get_inputs())

output = ort_sess.run(None, {"input": observation})

print(output)

# Check that the predictions are the same
# with th.no_grad():
#     print(model.policy(th.as_tensor(observation), deterministic=False))

"""