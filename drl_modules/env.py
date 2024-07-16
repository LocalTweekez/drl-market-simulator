import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import drl_modules.obs_spaces, drl_modules.data_extract, drl_modules.create_metadata
from drl_modules.data_extract import get_csv_path_pandas, extract_data, extract_data_windows
from drl_modules.obs_spaces import get_obs_space
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
                 dataset_path: str | pd.DataFrame  = "",
                 batch_size: int = 10,
                 init_balance: int = 10000,
                 risk_percentage: int = 2,
                 trading_fees: float = 0.01):
        super().__init__()

        print("Initializing trading environment")
        self.batch_size = batch_size
        self.init_balance = init_balance
        self.risk_percentage = risk_percentage
        self.symbol = symbol
        self.fees = trading_fees
        self.reward_func_idx = reward_func_idx

        if type(dataset_path) == str:
            # Init and process dataset
            if drl_modules.create_metadata.check_os == "win":
                self.df = extract_data_windows()
            else:
                # df_path = get_csv_path_pandas()
                self.df = extract_data(dataset_path)
        
        else:
            self.df = dataset_path

        # Action space: first value for long/short (-1 to 1), second value for risk management (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        
        # Observation space
        self.observation_space = get_obs_space(df=self.df, box_length=batch_size)

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

            # Check for exits and reset position properties if executed
            self.check_exit(direction=direction)

        # Placeholder: Update the state with actual logic based on the action
        self.state = self._get_observation()
        self.reward = self._calculate_reward()
        self.total_reward += self.reward
        self.done = self.t >= len(self.trade_env) - 1
        self.info = {
            "Step": self.t,
            "StepGlobal": self.t_global,
            "Balance": self.trade_env['eAccBalance'][self.t],
            "Action": direction,
            "Position": self.trade_env['pType'][self.t],
            "Reward": self.reward,
            "TotalReward": self.total_reward,
            "Done": self.done,
            "Wins": self.trade_env['eWins'][self.t],
            "Losses": self.trade_env['eLosses'][self.t],
            "TotalTrades": self.trade_env['eWins'][self.t] + self.trade_env['eLosses'][self.t]
        }

        return self.state, self.reward, self.done, self.done, self.info


    def render(self, mode='human', font_scaler=1.4, save_directory="", figure_name="trade_env_plot", plot_graph=True):
        history_df = pd.DataFrame(self.trade_env)

        # Create subplots with different heights, account balance chart smaller and below the price chart
        fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        y_ax1 = history_df["eAccBalance"]
        y_ax2 = history_df["close"]
        positions = history_df["pType"]
        times = history_df['time']

        # First chart: Price action with env action and positions (Placed as the first subplot now)
        colors = ['green' if pt == 1 else 'red' if pt == 2 else 'blue' for pt in positions]
        ax2.plot(times, y_ax2, label="Price movements with actions, RF: "+self.reward_func_name, color="blue")
        for i in range(1, len(times)):
            ax2.plot(times[i - 1:i + 1], y_ax2[i - 1:i + 1], color=colors[i - 1])

        prev_pos_type = history_df['pType'].shift(1).fillna(0)

        # Identify buy and sell initiation points
        buy_mask = (history_df['pType'] == 1) & (history_df['actDir'] > 0.5) & (prev_pos_type == 0)
        sell_mask = (history_df['pType'] == 2) & (history_df['actDir'] < -0.5) & (prev_pos_type == 0)

        # Win / Loss rate
        size = history_df["time"].size - 1
        wins = history_df["eWins"][size]
        losses = history_df["eLosses"][size]

        # Long / Short trades amount
        long_trades = sum((self.positions[i]["positionType"] == "BUY" for i in range(len(self.positions))))
        short_trades = sum((self.positions[i]["positionType"] == "SELL" for i in range(len(self.positions))))

        # Extract times and prices for plotting
        buy_times = times[buy_mask]
        sell_times = times[sell_mask]
        buy_prices = y_ax2[buy_mask]
        sell_prices = y_ax2[sell_mask]

        ax2.scatter(buy_times, buy_prices, color='green', label='Buy', marker='^', s=100, zorder=5)
        ax2.scatter(sell_times, sell_prices, color='red', label='Sell', marker='v', s=100, zorder=5)

        ax2.set_title(f"Price Movement with Trades ({self.symbol})", fontsize=18*font_scaler)
        ax2.set_ylabel("Price", fontsize=16*font_scaler)
        ax2.legend(loc='upper left', fontsize=14*font_scaler)

        # Adding a text box
        textstr = f"Wins: {wins}\nLosses: {losses}\nLong trades: {long_trades} Short trades: {short_trades}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # Corrected keyword 'facecolor'
        fig.gca().text(0.015, 0.84, textstr, transform=plt.gca().transAxes, fontsize=16*font_scaler,
                    verticalalignment='top', bbox=props)

        # Add open and close markers for each position
        for pos in self.positions:
            open_time = pos['timeOpen']
            close_time = pos['timeClose']  # Example close time
            open_price = pos['priceOpen']
            close_price = pos['priceClose']
            ax2.plot([open_time, close_time], [open_price, close_price], color='black', linestyle='--', linewidth=1)
            ax2.scatter(close_time, close_price, color='black', marker='o', s=50, zorder=5)

        # Second chart: Account balance over time (Placed as the second subplot and smaller)
        ax1.plot(times, y_ax1, label="Account balance", color="blue", linewidth=2)
        ax1.set_title('Account balance over time', fontsize=18*font_scaler)
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.set_xlabel("Time", fontsize=16*font_scaler)
        ax1.set_ylabel("Balance", fontsize=16*font_scaler)
        ax1.legend(loc='upper left', fontsize=14*font_scaler)

        plt.xticks(rotation=45, fontsize=12*font_scaler)  # Adjust rotation and font size for better readability
        plt.yticks(fontsize=12*font_scaler)  # Adjust font size for y-axis ticks
        plt.tight_layout()  # Adjust layout to make room for labels and title

        plt.savefig(f"{save_directory+figure_name}.png")
        history_df.to_csv(f"{save_directory+figure_name}.csv", index=False)

        plt.close(fig)

    def close(self):
        plt.close("all")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Special functions

    def check_exit(self, direction):
        diff_pips = 0
        tradetype = ""
        comment = ""
        exit = False
        sl_hit = False
        dud_trade = False
        limit = (0.5 + pow(0.5, self.position_total)) if self.position_total != 0 else 0.5

        if self.trade_env["pType"][self.t] == 1:
            tradetype = "LONG"
            if self.trade_env["close"][self.t-1] <= self.trade_env["pSLPrice"][self.t]:
                sl_hit = True
            elif self.trade_env["pOpenPrice"][self.t] == self.trade_env["close"][self.t-1]:
                dud_trade = True
            if direction < -limit or sl_hit:
                diff_pips = (self.trade_env["close"][self.t-1] - self.trade_env["pOpenPrice"][self.t-1])
                exit = True
        else:
            tradetype = "SHORT"
            if self.trade_env["close"][self.t-1] >= self.trade_env["pSLPrice"][self.t]:
                sl_hit = True
            elif self.trade_env["pOpenPrice"][self.t] == self.trade_env["close"][self.t-1]:
                dud_trade = True
            if direction > limit or sl_hit:
                diff_pips = (self.trade_env["pOpenPrice"][self.t-1] - self.trade_env["close"][self.t-1])
                exit = True
        
        # Return if no exit signal was given
        if not exit:
            return
        
        # Calculate reward
        profit_loss = self._calculate_profit_loss(diff_pips)

        # Add profit/loss to step
        self.trade_env["pProfit"][self.t] = (profit_loss - self.fees).__round__(2)
        self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
        if self.trade_env["pProfit"][self.t] > 0:
            self.trade_env["eWins"][self.t] += 1
        else:
            self.trade_env["eLosses"][self.t] += 1

        # Save position info into array
        if sl_hit:
            comment = f"({tradetype}) Stop loss hit - failed trade"
        elif dud_trade:
            comment = f"({tradetype}) Dud trade, no win nor loss (only commission costs)"
        else:
            comment = f"({tradetype}) Position closed by execution"

        position = Position(self.trade_env["pType"][self.t], 
        self.trade_env["pSLPrice"][self.t], 
        self.trade_env["pTPPrice"][self.t],
        self.timeOpen,
        self.trade_env["time"][self.t],
        self.trade_env["pOpenPrice"][self.t],
        self.trade_env["close"][self.t-1],
        self.trade_env["pProfit"][self.t],
        self.symbol,
        comment)
        self.positions.append(position.getPositionInfo())

        # Reset position properties
        self.trade_env["pType"][self.t] = 0
        self.trade_env["pOpenPrice"][self.t] = 0
        self.trade_env["pSLPrice"][self.t] = 0
        self.trade_env["pTPPrice"][self.t] = 0
    
    def _calculate_atr(self, period=14):
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.DataFrame({'tr': high_low, 'tr1': high_close, 'tr2': low_close}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        # Return the last ATR value
        return atr.iloc[-1] if not atr.empty else 0.0
    
    def _calculate_profit_loss(self, diff_pips, lot=10000):
        risk_per_trade = self.trade_env["eAccBalance"][self.t] * (self.risk_percentage/100)
        pip_value = (1/self.trade_env["close"][self.t-1]) * lot
        sl_pips = 1.5 * self._calculate_atr()
        lot_size = risk_per_trade / (sl_pips * pip_value)
        pip_multiplier = 10000
        profit_loss = diff_pips * pip_multiplier * lot_size
        return profit_loss

    def _get_observation(self):
        obs = {
            'open': self.trade_env['open'][self.t - self.batch_size:self.t],
            'high': self.trade_env['high'][self.t - self.batch_size:self.t],
            'low': self.trade_env['low'][self.t - self.batch_size:self.t],
            'close': self.trade_env['close'][self.t - self.batch_size:self.t],
            'volume': self.trade_env['tick_volume'][self.t - self.batch_size:self.t],
            'position': np.array([self.trade_env['pType'][self.t]], dtype=np.float32)  # Include the current position type
        }
        return obs

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