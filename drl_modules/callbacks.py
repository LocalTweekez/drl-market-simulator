import os
import glob
import requests
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import gym
from stable_baselines3 import PPO

import tkinter as tk
from tkinter.filedialog import askdirectory

def print_values(locals_dict):
    for key, value in locals_dict.items():
        print(f"{key}: {value}")
    print("\n")

class LoggingCallback(BaseCallback):
    def __init__(self, log_dir, dataset_size, remote_url, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_csv = os.path.join(log_dir, "info_log.csv")
        self.step_count = 0
        self.update_interval = 2048
        self.update_iteration = dataset_size - 1
        self.info = {}
        self.remote_url = remote_url
        self._initialize_csv()

    def _initialize_csv(self):
        df = pd.DataFrame(columns=['Step', 'StepGlobal', 'Balance', 'Action', 'Position', 'Reward', 'TotalReward', 'Done', 'Wins', 'Losses', 'TotalTrades', 'TimeLimit.truncated', 'Truncated'])
        df.to_csv(self.log_csv, index=False)

    def _log_to_csv(self, info):
        df = pd.DataFrame([info])
        df.to_csv(self.log_csv, mode='a', header=False, index=False)  # Write to CSV without an additional index column

    def _on_step(self) -> bool:
        self.info = self.locals["infos"][0].copy()  # Copy to avoid modifying the original
        self.info.pop('terminal_observation', None)  # Remove terminal_observation if it exists
        step = self.info["Step"]

        # Full dataset iteration check
        if step % self.update_iteration == 0:
            self._log_to_csv(self.info)
            # Send log to remote server

        return True

    def _on_rollout_end(self) -> None:
        # Pass the method without any action
        pass

    def _on_training_end(self) -> None:
        plot_total_rewards(self.log_dir)
        return True

    def send_log_to_server(self, log_info):
        try:
            response = requests.post(
                self.remote_url,
                json=log_info,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                print("Log successfully sent to server")
            else:
                print(f"Failed to send log to server, status code: {response.status_code}")
        except Exception as e:
            print(f"An error occurred while sending log to server: {e}")

class EventCallback(BaseCallback):
    """
    Base class for triggering callback on event.

    :param callback: Callback that will be called when an event is triggered.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, callback: BaseCallback, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        self.callback.parent = self

    def _on_step(self) -> bool:
        if self.event_trigger():
            return self.callback.on_step()
        return True

    def _on_rollout_end(self) -> None:
        if self.event_trigger():
            self.callback.on_rollout_end()

    def event_trigger(self) -> bool:
        # Define the condition for triggering the event
        # For example, you can check if a certain number of steps have been reached
        return self.num_timesteps % self.locals['self'].n_steps == 0

def plot_total_rewards(csv_file_dir):
    if csv_file_dir == "":
        root = tk.Tk()
        root.withdraw()
        csv_file_dir = askdirectory(
            title="Select directory with csv file"
        )
        csv_file_dir += "/"

    df = pd.read_csv(csv_file_dir + "info_log.csv")
    plt.figure(figsize=(10, 6))

    x_arr = df["StepGlobal"].tolist()
    y_arr = df["TotalReward"].tolist()

    plt.plot(x_arr, y_arr, marker='o', linestyle='-')
    plt.title('Total Reward Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(csv_file_dir + "iteration_total_reward.png")
    plt.close()


def plot_loss_accuracy(run_dir: str) -> None:
    """Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    run_dir: str
        Directory containing training ``info_log.csv`` and optional
        ``eval_results.csv`` produced by ``ppo_eval``.
    """

    train_files = sorted(glob.glob(os.path.join(run_dir, "part*/info_log.csv")))
    if not train_files:
        single = os.path.join(run_dir, "info_log.csv")
        if os.path.exists(single):
            train_files = [single]
    if not train_files:
        return

    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

    train_loss = -train_df["TotalReward"].tolist()
    trades = train_df["TotalTrades"].replace(0, pd.NA)
    train_acc = (train_df["Wins"] / trades).fillna(0).tolist()

    eval_file = os.path.join(run_dir, "eval_results.csv")
    val_loss = val_acc = None
    if os.path.exists(eval_file):
        val_df = pd.read_csv(eval_file)
        val_loss = (-val_df["total_reward"]).tolist()
        val_acc = (val_df["win_rate"] / 100).tolist()

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(train_loss, label="train")
    if val_loss is not None:
        ax_loss.plot(val_loss, label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Episode")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(train_acc, label="train")
    if val_acc is not None:
        ax_acc.plot(val_acc, label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Episode")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_accuracy.png"))
    plt.close(fig)

def main():
    log_dir = "path_to_log_dir"
    dataset_size = 1000  # Example dataset size
    remote_url = "https://your-server.com/api/logs"  # Replace with your actual server URL

    logging_callback = LoggingCallback(log_dir, dataset_size, remote_url)
    event_callback = EventCallback(callback=logging_callback)

    # Example usage with a Stable Baselines 3 model
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, callback=event_callback)

    # Plotting the total rewards after training
    csv_file_dir = log_dir
    plot_total_rewards(csv_file_dir)

if __name__ == "__main__":
    plot_total_rewards("")
