import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
import time
from stable_baselines3.ppo import MlpPolicy

from drl_modules.rewards import RewardFunctions

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from gymnasium import spaces
from stable_baselines3.ppo import MlpPolicy


class CustomCNNPolicyNet(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomCNNPolicyNet, self).__init__(observation_space, features_dim)
        
        # Define the CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1),  # Adjust kernel size and padding if needed
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size after CNN layers
        with th.no_grad():
            dummy_input = th.as_tensor(observation_space.sample()[None]).float()
            dummy_input = dummy_input.unsqueeze(1)  # Adding a channel dimension
            n_flatten = self.cnn(dummy_input).shape[1]
        
        # Define the fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(1)  # Add a channel dimension if it's missing
        return self.linear(self.cnn(observations))

    def save_model(self, file_path: str):
        """Save the model to a file."""
        th.save(self.state_dict(), file_path)

    @classmethod
    def load_model(cls, file_path: str, observation_space, features_dim):
        """Load the model from a file."""
        model = cls(observation_space, features_dim)
        model.load_state_dict(th.load(file_path))
        return model
    

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super(LSTMFeatureExtractor, self).__init__(observation_space, features_dim)
        self.hidden_size = 64
        self.num_layers = 4
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        n_input_channels = observation_space.shape[0]  # Adjusted to match the correct dimension
        self.lstm = nn.LSTM(n_input_channels, self.hidden_size, self.num_layers, 
                            batch_first=True, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Reshape observations to (batch_size, seq_len, n_input_channels)
        # Assuming observations are (batch_size, seq_len, n_input_channels)
        observations = observations.permute(0, 2, 1)
        h0 = th.zeros(self.num_layers * self.num_directions, observations.size(0), self.hidden_size).to(observations.device)
        c0 = th.zeros(self.num_layers * self.num_directions, observations.size(0), self.hidden_size).to(observations.device)
        lstm_out, _ = self.lstm(observations, (h0, c0))
        features = self.fc(lstm_out[:, -1, :])
        return features


class CustomLSTMMlpPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        # Define custom feature extractor
        kwargs['features_extractor_class'] = LSTMFeatureExtractor
        kwargs['features_extractor_kwargs'] = {'features_dim': 128}
        super(CustomLSTMMlpPolicy, self).__init__(*args, **kwargs)
