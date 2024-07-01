import gym
from gym import spaces
import numpy as np
import pandas as pd
from drl_modules import obs_spaces, data_extract, create_metadata
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class TradingEnv(gym.Env):
    def __init__(self,
                 batch_size: int = 10,
                 init_balance: int = 10000,
                 risk_percentage: int = 2,
                 trading_fees: float = 0.01):
        super().__init__()

        print("Initializing trading environment")
        self.batch_size = batch_size
        self.init_balance = init_balance
        self.risk_percentage = risk_percentage
        self.fees = trading_fees

        # Init and process dataset
        if create_metadata.check_os == "win":
            self.df = data_extract.extract_data_windows()
        else:
            self.df = data_extract.extract_data()
        
        # Action space: first value for long/short (-1 to 1), second value for risk management (0 to 1)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        
        # Observation space
        self.observation_space = obs_spaces.get_obs_space(df=self.df, box_length=batch_size)

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

        # Environment parameters init
        self.t = self.batch_size  # Initialize t to batch_size
        self.reset()
    
    def reset(self):
        self.t = self.batch_size  # Reset t to batch_size
        self.state = self._get_observation()
        self.done = False
        self.reward = 0
        self.info = {}
        return self.state
    
    def step(self, action):
        if self.t >= len(self.df) - 1:
            self.done = True
            self._save_trade_env_to_csv()
            return self.state, self.reward, self.done, self.info
        
        self.t += 1  # Increment t

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
        risk_factor = 1.5 * self.calculate_atr() * risk
        if self.trade_env["pType"][self.t-1] == 0:
            if direction > 0.5:
                self.trade_env["pType"][self.t] = 1
                self.trade_env["pOpenPrice"][self.t] = self.trade_env["close"][self.t]
                self.trade_env["pSLPrice"][self.t] = self.trade_env["close"][self.t] - risk_factor
            elif direction < -0.5:
                self.trade_env["pType"][self.t] = 2
                self.trade_env["pOpenPrice"][self.t] = self.trade_env["close"][self.t]
                self.trade_env["pSLPrice"][self.t] = self.trade_env["close"][self.t] + risk_factor
            else:
                self.trade_env["pType"][self.t] = 0

        # If there is a position active
        else:
            # Carry over 
            self.trade_env["pType"][self.t] = self.trade_env["pType"][self.t-1]
            self.trade_env["pOpenPrice"][self.t] = self.trade_env["pOpenPrice"][self.t-1]
            self.trade_env["pSLPrice"][self.t] = self.trade_env["pSLPrice"][self.t-1]
            self.trade_env["pTPPrice"][self.t] = self.trade_env["pTPPrice"][self.t-1]

            # Check for exits and reset position properties if executed
            self.check_exit(action=direction)

        # Placeholder: Update the state with actual logic based on the action
        self.state = self._get_observation()
        self.reward = self._calculate_reward(action)
        self.done = self.t >= len(self.trade_env) - 1
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def _save_trade_env_to_csv(self):
        df = pd.DataFrame(self.trade_env)
        df.to_csv("trade_env.csv", index=False)
        print("SAVED TO CSV FILE!!!!!!!!!!!!!")

    def render(self, mode='human', font_scaler=1.4):

        history_df = pd.DataFrame(self.trade_env)

        # Create subplots with different heights, account balance chart smaller and below the price chart
        fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        y_ax1 = history_df["eAccBalance"]
        y_ax2 = history_df["close"]
        positions = history_df["pType"]
        times = history_df['time']

        # First chart: Price action with env action and positions (Placed as the first subplot now)
        colors = ['green' if pt == 1 else 'red' if pt == 2 else 'blue' for pt in positions]
        ax2.plot(times, y_ax2, label="Price movements with actions", color="blue")
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

        # Extract times and prices for plotting
        buy_times = times[buy_mask]
        sell_times = times[sell_mask]
        buy_prices = y_ax2[buy_mask]
        sell_prices = y_ax2[sell_mask]

        ax2.scatter(buy_times, buy_prices, color='green', label='Buy', marker='^', s=100, zorder=5)
        ax2.scatter(sell_times, sell_prices, color='red', label='Sell', marker='v', s=100, zorder=5)

        ax2.set_title("Price Movement with Trades", fontsize=18*font_scaler)
        ax2.set_ylabel("Price", fontsize=16*font_scaler)
        ax2.legend(loc='upper left', fontsize=14*font_scaler)

        # Adding a text box
        long_trades = np.sum((self.trade_env['pType'] == 1) & (self.trade_env['actDir'] > 0.5))
        short_trades = np.sum((self.trade_env['pType'] == 2) & (self.trade_env['actDir'] < -0.5))
        textstr = f"Wins: {wins}\nLosses: {losses}\nLong trades: {long_trades} Short trades: {short_trades}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # Corrected keyword 'facecolor'
        fig.gca().text(0.015, 0.84, textstr, transform=plt.gca().transAxes, fontsize=16*font_scaler,
                    verticalalignment='top', bbox=props)

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

        figure_name="trade_env_plot"
        print("Saving figure {}".format(figure_name))
        plt.savefig(f"{figure_name}.png")
        history_df.to_csv(f"{figure_name}.csv", index=False)

        plt.close(fig)


    def close(self):
        plt.close("all")

    # Special functions

    def check_exit(self, action):
        diff_pips = 0
        exit = False

        if self.trade_env["pType"][self.t] == 1:
            if action < -0.5:
                diff_pips = (self.trade_env["close"][self.t-1] - self.trade_env["pOpenPrice"][self.t-1]) * 10000
                exit = True
        else:
            if action > 0.5:
                diff_pips = (self.trade_env["pOpenPrice"][self.t-1] - self.trade_env["close"][self.t-1]) * 10000
                exit = True
        
        # Return if no exit signal was given
        if not exit:
            return
        
        # Add profit/loss to step
        diff = diff_pips * 10  # times pip value 
        self.trade_env["pProfit"][self.t] = diff - self.fees
        self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
        if self.trade_env["pProfit"][self.t] > 0:
            self.trade_env["eWins"][self.t] += 1
        else:
            self.trade_env["eLosses"][self.t] += 1

        # Reset position properties
        self.trade_env["pType"][self.t] = 0
        self.trade_env["pOpenPrice"][self.t] = 0
        self.trade_env["pSLPrice"][self.t] = 0
        self.trade_env["pTPPrice"][self.t] = 0
    
    def calculate_atr(self, period=14):
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        tr = pd.DataFrame({'tr': high_low, 'tr1': high_close, 'tr2': low_close}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        # Return the last ATR value
        return atr.iloc[-1] if not atr.empty else 0.0

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

    def _calculate_reward(self, action):
        # Placeholder: Implement your reward calculation logic
        reward = np.random.random()  # Replace with actual reward logic
        return reward


if __name__ == "__main__":
    tradingenv = TradingEnv()
    state = tradingenv.reset()
    print("Initial State:", state)

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
            tradingenv._save_trade_env_to_csv()
            break
    
    tradingenv.render()
