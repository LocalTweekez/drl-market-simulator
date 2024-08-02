import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import drl_modules.obs_spaces, drl_modules.data_extract, drl_modules.create_metadata
from drl_modules.data_extract import get_csv_path_pandas, extract_data, extract_data_windows
from drl_modules.obs_spaces import get_obs_space_dict, get_obs_space_flattened
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from drl_modules.rewards import RewardFunctions
import datetime
from gymnasium.utils import seeding

class Position:
    id = 1

    def __init__(self, 
                 positionType: int,
                 positionSL: float,
                 positionTP: float,
                 timeOpen: datetime.datetime,
                 timeClose: datetime.datetime,
                 priceOpen: float, 
                 priceClose: float,
                 profit: float,
                 symbol: str,
                 comment: str):
        self.id = Position.id
        Position.id += 1

        self.positionType = "BUY" if positionType == 1 else "SELL"
        self.positionSL = positionSL
        self.positionTP = positionTP
        self.timeOpen = timeOpen
        self.timeClose = timeClose
        self.priceOpen = priceOpen
        self.priceClose = priceClose
        self.tradeResult = "null"
        self.profit = profit
        self.symbol = symbol
        self.comment = comment

        if self.positionType == "BUY":
            if self.priceClose > self.priceOpen:
                self.tradeResult = "WIN"
            else:
                self.tradeResult = "LOSS"
        else:
            if self.priceClose < self.priceOpen:
                self.tradeResult = "WIN"
            else:
                self.tradeResult = "LOSS"

    def getPositionInfo(self):
        return {
            'id': self.id,
            'timeOpen': self.timeOpen,
            'timeClose': self.timeClose,
            'positionType': self.positionType,
            'tradeResult': self.tradeResult,
            'priceOpen': self.priceOpen,
            'priceClose': self.priceClose,
            'positionSL': self.positionSL,
            'positionTP': self.positionTP,
            'profit': self.profit,
            'symbol': self.symbol,
            'comment': self.comment
        }


class TradingEnv(gym.Env):
    def __init__(self,
                 reward_func_idx: int,
                 symbol: str,
                 agent_policy: str,
                 dataset_path: str | pd.DataFrame  = "",
                 batch_size: int = 10,
                 init_balance: int = 10000,
                 risk_percentage: int = 2,
                 trading_fees: float = 0.01,
                 normalize: bool = False):  # Added normalize parameter
        super().__init__()

        print("Initializing trading environment")
        self.batch_size = batch_size
        self.init_balance = init_balance
        self.risk_percentage = risk_percentage
        self.symbol = symbol
        self.fees = trading_fees
        self.reward_func_idx = reward_func_idx
        self.agent_policy = agent_policy
        self.normalize = normalize  # Store normalize parameter

        if type(dataset_path) == str:
            # Init and process dataset
            if drl_modules.create_metadata.check_os == "win":
                self.df = extract_data_windows()
            else:
                self.df = extract_data(dataset_path)
        
        else:
            self.df = dataset_path

        # Normalize dataset if normalize parameter is True
        if self.normalize:
            self.df = self._normalize_dataframe(self.df)

        # Action space: first value for long/short (-1 to 1), second value for risk management (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        
        # Observation space
        self.observation_space = get_obs_space_flattened(df=self.df, box_length=batch_size) if agent_policy != "MultiInputPolicy" else get_obs_space_dict(df=self.df, box_length=batch_size)

        # Environment data init
        dtype = [
            ('time', 'datetime64[m]'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('tick_volume', 'i4'),
            ('actDir', 'f4'),
            ('actRisk', 'f4'),
            ('eReward', 'f8'),
            ('eAccBalance', 'f8'),
            ('eWins', 'i4'),
            ('eLosses', 'i4'),
            ('pType', 'i4'),
            ('pOpenPrice', 'f8'),
            ('pSLPrice', 'f8'),
            ('pTPPrice', 'f8'),
            ('pProfit', 'f8')
        ]
        size = self.df.shape[0]
        self.trade_env = np.zeros(size, dtype=dtype)

        # Copy data from df to trade_env
        for field in ['time', 'open', 'high', 'low', 'close', 'tick_volume']:
            self.trade_env[field] = self.df[field].values
        self.trade_env['eAccBalance'].fill(self.init_balance)

        # Init reward function
        self.rewardfunc = RewardFunctions()
        self.reward_func_name = self.rewardfunc.function_names[self.reward_func_idx]
        print("Reward function: ",self.reward_func_name, "(Reward index: ",self.reward_func_idx,")")

        # Environment parameters init
        self.t = self.batch_size  # Initialize t to batch_size
        self.t_global = self.t
        self.df_size = len(self.df)
        self.positions = []
        self.symbol = symbol
        self.position_total = 0
        self.total_reward = 0
        self.timeOpen = datetime.datetime.now()
        self.reset()

    
    def reset(self, seed=None):
        self.t = self.batch_size  # Reset t to batch_size
        self.state = self._get_observation()
        self.done = False
        self.reward = 0
        self.position_total = 0
        self.total_reward = 0
        self.timeOpen = datetime.datetime.now()
        self.info = {}
        self.positions = []
        self.reward_func_name = ""
        return self.state, {}
    
    def step(self, action):
        if self.t >= len(self.df) - 1:
            self.done = True
            return self.state, self.reward, self.done, self.done, self.info
        
        self.t += 1  # Increment t
        self.t_global += 1

        # Extract actions
        direction = action[0]  # Long/short decision (-1 to 1)
        risk = action[1]       # Risk management (0 to 1)
        
        # Step variables
        price = self.trade_env["close"][self.t-1]
        last_pos = self.trade_env["pType"][self.t-1]
        last_bal = self.trade_env["eAccBalance"][self.t-1]
        self.trade_env["pProfit"][self.t] = 0
        self.trade_env["actDir"][self.t] = direction
        self.trade_env["actRisk"][self.t] = direction
        self.trade_env["pType"][self.t] = last_pos
        self.trade_env["eAccBalance"][self.t] = last_bal
        self.trade_env["eWins"][self.t] = self.trade_env["eWins"][self.t-1]
        self.trade_env["eLosses"][self.t] = self.trade_env["eLosses"][self.t-1]


        # If no position exists
        neural_risk = 0.5 + risk # 
        risk_factor = 1.5 * self._calculate_atr() * neural_risk
        if self.trade_env["pType"][self.t-1] == 0:
            self.position_total = 0
            if direction > 0.5:
                self.trade_env["pType"][self.t] = 1
                self.trade_env["pOpenPrice"][self.t] = self.trade_env["close"][self.t]
                self.trade_env["pSLPrice"][self.t] = (self.trade_env["close"][self.t] - risk_factor).__round__(4)
                self.timeOpen = self.trade_env["time"][self.t]
            elif direction < -0.5:
                self.trade_env["pType"][self.t] = 2
                self.trade_env["pOpenPrice"][self.t] = self.trade_env["close"][self.t]
                self.trade_env["pSLPrice"][self.t] = (self.trade_env["close"][self.t] + risk_factor).__round__(4)
                self.timeOpen = self.trade_env["time"][self.t]
            else:
                self.trade_env["pType"][self.t] = 0

        # If there is a position active
        else:
            self.position_total += 1
            # Carry over 
            self.trade_env["pType"][self.t] = self.trade_env["pType"][self.t-1]
            self.trade_env["pOpenPrice"][self.t] = self.trade_env["pOpenPrice"][self.t-1]
            self.trade_env["pSLPrice"][self.t] = self.trade_env["pSLPrice"][self.t-1]
            self.trade_env["pTPPrice"][self.t] = self.trade_env["pTPPrice"][self.t-1]
            self.trade_env["pProfit"][self.t] = 0
            self.trade_env["eAccBalance"][self.t] = self.trade_env["eAccBalance"][self.t-1]
            self.trade_env["eWins"][self.t] = self.trade_env["eWins"][self.t-1]
            self.trade_env["eLosses"][self.t] = self.trade_env["eLosses"][self.t-1]

            # Close active position if SL or TP conditions are met
            if self.trade_env["pType"][self.t] == 1:  # Buy
                if self.trade_env["close"][self.t] <= self.trade_env["pSLPrice"][self.t]:
                    self.trade_env["pProfit"][self.t] = (self.trade_env["pSLPrice"][self.t] - self.trade_env["pOpenPrice"][self.t]) * (self.trade_env["eAccBalance"][self.t-1]/self.trade_env["pOpenPrice"][self.t])
                    self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
                    self.trade_env["eLosses"][self.t] += 1
                    self.done = True
                    self.positions.append(Position(positionType=1,
                                                   positionSL=self.trade_env["pSLPrice"][self.t],
                                                   positionTP=self.trade_env["pTPPrice"][self.t],
                                                   timeOpen=self.timeOpen,
                                                   timeClose=self.trade_env["time"][self.t],
                                                   priceOpen=self.trade_env["pOpenPrice"][self.t],
                                                   priceClose=self.trade_env["pSLPrice"][self.t],
                                                   profit=self.trade_env["pProfit"][self.t],
                                                   symbol=self.symbol,
                                                   comment="Loss - Stop Loss"))

                elif self.trade_env["close"][self.t] >= self.trade_env["pTPPrice"][self.t]:
                    self.trade_env["pProfit"][self.t] = (self.trade_env["pTPPrice"][self.t] - self.trade_env["pOpenPrice"][self.t]) * (self.trade_env["eAccBalance"][self.t-1]/self.trade_env["pOpenPrice"][self.t])
                    self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
                    self.trade_env["eWins"][self.t] += 1
                    self.done = True
                    self.positions.append(Position(positionType=1,
                                                   positionSL=self.trade_env["pSLPrice"][self.t],
                                                   positionTP=self.trade_env["pTPPrice"][self.t],
                                                   timeOpen=self.timeOpen,
                                                   timeClose=self.trade_env["time"][self.t],
                                                   priceOpen=self.trade_env["pOpenPrice"][self.t],
                                                   priceClose=self.trade_env["pTPPrice"][self.t],
                                                   profit=self.trade_env["pProfit"][self.t],
                                                   symbol=self.symbol,
                                                   comment="Win - Take Profit"))

            elif self.trade_env["pType"][self.t] == 2:  # Sell
                if self.trade_env["close"][self.t] >= self.trade_env["pSLPrice"][self.t]:
                    self.trade_env["pProfit"][self.t] = (self.trade_env["pOpenPrice"][self.t] - self.trade_env["pSLPrice"][self.t]) * (self.trade_env["eAccBalance"][self.t-1]/self.trade_env["pOpenPrice"][self.t])
                    self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
                    self.trade_env["eLosses"][self.t] += 1
                    self.done = True
                    self.positions.append(Position(positionType=2,
                                                   positionSL=self.trade_env["pSLPrice"][self.t],
                                                   positionTP=self.trade_env["pTPPrice"][self.t],
                                                   timeOpen=self.timeOpen,
                                                   timeClose=self.trade_env["time"][self.t],
                                                   priceOpen=self.trade_env["pOpenPrice"][self.t],
                                                   priceClose=self.trade_env["pSLPrice"][self.t],
                                                   profit=self.trade_env["pProfit"][self.t],
                                                   symbol=self.symbol,
                                                   comment="Loss - Stop Loss"))

                elif self.trade_env["close"][self.t] <= self.trade_env["pTPPrice"][self.t]:
                    self.trade_env["pProfit"][self.t] = (self.trade_env["pOpenPrice"][self.t] - self.trade_env["pTPPrice"][self.t]) * (self.trade_env["eAccBalance"][self.t-1]/self.trade_env["pOpenPrice"][self.t])
                    self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
                    self.trade_env["eWins"][self.t] += 1
                    self.done = True
                    self.positions.append(Position(positionType=2,
                                                   positionSL=self.trade_env["pSLPrice"][self.t],
                                                   positionTP=self.trade_env["pTPPrice"][self.t],
                                                   timeOpen=self.timeOpen,
                                                   timeClose=self.trade_env["time"][self.t],
                                                   priceOpen=self.trade_env["pOpenPrice"][self.t],
                                                   priceClose=self.trade_env["pTPPrice"][self.t],
                                                   profit=self.trade_env["pProfit"][self.t],
                                                   symbol=self.symbol,
                                                   comment="Win - Take Profit"))

        # Assign reward based on function
        self.reward = getattr(self.rewardfunc, self.reward_func_name)(self.trade_env, self.t)
        
        self.state = self._get_observation()
        return self.state, self.reward, self.done, self.done, self.info
    
    def _get_observation(self):
        """Returns the current observation after applying normalization if required."""
        obs = self.df.iloc[self.t - self.batch_size:self.t].values
        if self.normalize:
            obs = self._normalize_observation(obs)
        return obs
    
    def _normalize_dataframe(self, df):
        """Normalize the entire dataframe."""
        for col in df.columns:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df
    
    def _normalize_observation(self, obs):
        """Normalize the observation."""
        # Assuming `obs` is a 2D array where each row represents a timestamp and each column represents a feature.
        for i in range(obs.shape[1]):
            feature = obs[:, i]
            mean = feature.mean()
            std = feature.std()
            if std > 0:
                obs[:, i] = (feature - mean) / std
            else:
                obs[:, i] = feature - mean
        return obs
    
    def _calculate_atr(self):
        """Calculate Average True Range (ATR)."""
        df = self.df[['high', 'low', 'close']]
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(np.abs(df['high'] - df['prev_close']), 
                                         np.abs(df['low'] - df['prev_close'])))
        atr = df['tr'].rolling(window=14).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0.01

    
    def _calculate_profit_loss(self, diff_pips, lot = 10000):
        risk_per_trade = self.trade_env["eAccBalance"][self.t] * (self.risk_percentage/100)
        pip_value = (1/self.trade_env["close"][self.t-1]) * lot
        sl_pips = 1.5 * self._calculate_atr()
        lot_size = risk_per_trade / (sl_pips * pip_value)
        pip_multiplier = 10000
        profit_loss = diff_pips * pip_multiplier * lot_size
        return profit_loss

    def _get_observation(self):
        if self.agent_policy == "MultiInputPolicy":
            return {
                'open': self.trade_env['open'][self.t-self.batch_size : self.t],
                'high': self.trade_env['high'][self.t-self.batch_size : self.t],
                'low': self.trade_env['low'][self.t-self.batch_size : self.t],
                'close': self.trade_env['close'][self.t-self.batch_size : self.t],
                'volume': self.trade_env['tick_volume'][self.t-self.batch_size : self.t],
                'position': self.trade_env['pType'][self.t-self.batch_size : self.t]  # Include the current position type
            }
        else:
            open_prices = self.trade_env['open'][self.t - self.batch_size: self.t]
            high_prices = self.trade_env['high'][self.t - self.batch_size: self.t]
            low_prices = self.trade_env['low'][self.t - self.batch_size: self.t]
            close_prices = self.trade_env['close'][self.t - self.batch_size: self.t]
            volumes = self.trade_env['tick_volume'][self.t - self.batch_size: self.t]
            positions = self.trade_env['pType'][self.t - self.batch_size: self.t]  # Include the current position type

            # Flatten the observation
            return np.concatenate([
                open_prices,
                high_prices,
                low_prices,
                close_prices,
                volumes,
                positions
            ]).astype(np.float32)

    def _calculate_reward(self):
        reward = self.rewardfunc[self.reward_func_idx](self.trade_env, self.t)
        self.reward_func_name = self.rewardfunc.function_names[self.reward_func_idx]

        return reward
    
    def _save_trade_env_to_csv(self, save_directory=""):
        df = pd.DataFrame(self.trade_env)
        df.to_csv(save_directory+"trade_env.csv", index=False)

    def _save_positions_into_csv(self, save_directory=""):
        columns = ['id', 'timeOpen', 'timeClose', 'positionType', 'tradeResult', 'priceOpen', 'priceClose', 'positionSL', 'positionTP', 'profit', 'symbol', 'comment']
        pos_df = pd.DataFrame(self.positions, columns=columns)
        pos_df.to_csv(save_directory+"positions.csv", index=False)
        print("SAVED ALL POSITIONS")

    def _get_env_details(self):
        details = {
            'Symbol': self.symbol,
            'Dataset Size': self.df_size,
            'Initial Balance': self.init_balance,
            'Risk Percentage': self.risk_percentage,
            'Trading Fees': self.fees,
            'Reward Function Index': self.reward_func_idx,
            'Reward Function Name': self.reward_func_name,
            'Batch Size': self.batch_size,
            'Action Space': self.action_space,
            'Observation Space': self.observation_space,
            'Dataset Columns': list(self.df.columns),
        }
        
        print("\nEnvironment Details:")
        for key, value in details.items():
            print(f"{key}: {value}")
        print()

        return details


def test():
    tradingenv = TradingEnv()
    state = tradingenv.reset()
    tmp_path= "../results/"
    
    c = 0
    while True:
        c += 1
        action = tradingenv.action_space.sample()  # Sample random action
        # print("ACTION: ", action[0])
        state, reward, done, info = tradingenv.step(action)
        # tradingenv.render()
        if c % 1000 == 0:
            print(f"{c}")
            
        if done:
            tradingenv._save_trade_env_to_csv(save_directory=tmp_path)
            tradingenv._save_positions_into_csv(save_directory=tmp_path)
            break
    
    tradingenv.render(save_directory=tmp_path)


if __name__ == "__main__":
    env = TradingEnv(0, "EURUSD")
    env._get_env_details()