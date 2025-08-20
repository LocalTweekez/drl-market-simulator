# DRL FinTech

A framework for experimenting with deep reinforcement learning for algorithmic trading. It provides a gym-based trading environment, training/evaluation utilities built on Stable-Baselines3 algorithms (PPO, A2C and DQN), and tools for exporting trained agents to ONNX.

## Repository structure

- `drl_modules/` – core package containing the environment, training helpers, reward functions and export utilities.
- `datasets/` – sample market data CSV files.
- `misc/` – assorted scripts and artifacts produced during experiments.
- `results/` – default output directory for trained models, logs and plots.
- `main.py` – command-line entry point that orchestrates training and evaluation.
- `onnx_example.py` – minimal example of exporting a trained model to ONNX.
- `requirements.txt` – Python dependencies.

## Installation

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The project targets Python 3.12 and uses Stable-Baselines3.

## Usage

Run the interactive driver to train and evaluate a reinforcement learning agent:

```bash
python main.py
```

You will be prompted for:

- dataset selection from `datasets/`
- reward function index
- currency symbol
- number of training batches and steps
- number of vectorized environments
- compute device (`cpu` or `cuda`)
- policy architecture (`MlpPolicy`, `CnnPolicy`, or `MultiInputPolicy`)
- learning algorithm (`PPO`, `A2C`, or `DQN`)
- output folder for results

The script trains the selected algorithm and then evaluates it, saving logs, plots and the model (e.g. `PPO_model.zip`) under the chosen results directory. A `configuration.yaml` file including the algorithm name is written alongside the outputs and can be reused to run evaluation-only sessions.

Example configuration:

```
Algorithm: PPO
Policy: MlpPolicy
...
```

## Exporting models

`drl_modules/export_model.py` converts a trained PPO policy to ONNX for inference outside Python. Both standard (flat) and dictionary-based observation spaces are supported.

## Datasets

CSV files must contain at least the following columns:

- `time`
- `open`
- `high`
- `low`
- `close`
- `tick_volume`

If these headers are missing, `drl_modules.data_extract.convert_csv_format` attempts to reformat the file. See `datasets/README.md` for more details.

## Results and logs

Training runs store models, TensorBoard logs, CSV trade histories and rendered plots under the results directory. Example output files can be found in `misc/`.

## License

No license information is provided.
