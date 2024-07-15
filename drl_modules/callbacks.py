import os
from stable_baselines3.common.callbacks import BaseCallback

"""
self.locals:
dict_keys(['self', 'total_timesteps', 'callback', 'log_interval', 'tb_log_name', 'reset_num_timesteps', 'progress_bar', 'iteration', 
        'env', 'rollout_buffer', 'n_rollout_steps', 'n_steps', 'obs_tensor', 'actions', 'values', 'log_probs', 'clipped_actions', 'new_obs', 
        'rewards', 'dones', 'infos', 'idx', 'done'])
Step: 6144, 
Info: {'Step': 6154, 'Balance': 12149.650000000014, 'Action': -1.0, 'Position': 2, 'Reward': 0.0, '
        TotalReward': 0.6552489758690465, 'Done': False, 'Wins': 509, 'Losses': 640, 'TotalTrades': 1149, 'TimeLimit.truncated': False}

"""


def print_values(locals_dict):
    for key, value in locals_dict.items():
        print(f"{key}: {value}")
    print("\n")


class LoggingCallback(BaseCallback):
    def __init__(self, log_dir, dataset_size, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_step = os.path.join(log_dir, "step_log.txt")
        self.log_rollout = os.path.join(log_dir, "rollout_log.txt")
        self.step_count = 0
        self.update_interval = 2048
        self.update_iteration = dataset_size-1
        self.info = {}

    def _on_step(self) -> bool:
        self.info = self.locals["infos"][0]
        step = self.info["Step"]

        # full dataset iteration check
        if step % self.update_iteration == 0:
            print(self.info)
            with open(self.log_step, "a") as f:
                f.write(f"{self.locals['infos'][0]}\n")


        return True

    def _on_rollout_end(self) -> None:
        with open(self.log_rollout, "a") as f:
            info = self.locals["infos"][0]
            log_info = (
                f"Step: {self.num_timesteps}, "
                f"Info: {info}"
            )
            f.write(log_info + "\n")
            # print(log_info)  # For debugging purposes
        return True

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
