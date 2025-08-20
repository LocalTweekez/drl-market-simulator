# drl_modules

Core modules used by the DRL FinTech project. The package implements the trading environment, training logic and supporting utilities.

## Contents


- `env.py` – Gymnasium environment representing a trading session. Supports continuous and discrete action spaces (`discrete_actions=True`). In discrete mode the actions map to: 0 hold, 1 buy, 2 sell.
- `ppo.py` – helpers for training and evaluating PPO agents.
- `dqn.py` – analogous helpers for Deep Q-Network agents using the discrete action space.
- `callbacks.py` – custom logging callbacks and plotting utilities.
- `data_extract.py` – functions for loading CSV datasets and splitting them into batches.
- `input_config.py` – interactive helpers for collecting run parameters and writing `configuration.yaml`.
- `export_model.py` – utilities for exporting trained models to ONNX and validating the exported graph.
- `rewards.py` – collection of reward functions selectable by index.
- `obs_spaces.py` – constructors for observation spaces in flat or dictionary format.
- `policies.py` – custom CNN and LSTM feature extractors for Stable-Baselines3 policies.
- `agent_render.py`, `eval_graph.py`, `verify_trade_env.py` – visualization and debugging helpers.

Each module is designed to be imported individually; `main.py` demonstrates how they fit together in a complete training workflow.
