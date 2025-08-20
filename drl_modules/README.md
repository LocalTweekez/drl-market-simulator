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

### Reward functions

`rewards.py` bundles several reward functions that can be selected by index when creating the environment.  They cover a range of trading objectives:

- `reward_main` – raw percentage change in account balance between steps.
- `reward_normalized` – percentage change scaled to ``[-1, 1]``.
- `reward_percentage_change` – portfolio percentage change using balance history.
- `reward_profit_loss` – absolute difference in portfolio valuation.
- `reward_sharpe_ratio` – Sharpe ratio of returns up to the current step.
- `reward_closing_high` – proximity of the close price to the period high.
- `reward_volume_weighted` – price movement weighted by traded volume.
- `reward_log_return` – logarithmic return of the portfolio balance.
- `reward_binary_profit` – binary outcome, 1 for profit and 0 for loss.
- `reward_compound_growth` – compound growth rate relative to the initial balance.
- `reward_environment_two` – step-wise percentage change in total balance.
- `reward_simplistic_comp` – compares balance to eight steps ago to capture momentum.
- `reward_with_drawdown` – balance change penalized by drawdown from the peak.
- `reward_combined` – Sharpe ratio minus maximum drawdown as a combined score.
- `reward_winrate` – deviation of current win rate from a target threshold.
- `test_sellonly` – toy function rewarding sells and penalizing buys.
- `reward_percentage_of_init` – percentage change relative to the initial balance.
- `reward_growth_trajectory` – moving-average growth of balance over a window.
