import gym
from gym import spaces
import numpy as np
import pandas as pd
from drl_modules import obs_spaces, data_extract, create_metadata

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
        
        # Action space
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 2]), dtype=np.float32)
        
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
            ('eAction', 'i4'),
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
        self.t = self.batch_size  # Initialize i to batch_size
        self.reset()
    
    def reset(self):
        self.t = self.batch_size  # Reset i to batch_size
        self.state = self._get_observation()
        return self.state
    
    def step(self, action):
        self.t += 1  # Increment i
        if self.t >= len(self.df):
            self.done = True
            return self.state, self.reward, self.done, self.info
        
        # Step variables
        price = self.trade_env["close"][self.t-1]
        last_pos = self.trade_env["pType"][self.t-1]
        last_bal = self.trade_env["eAccBalance"][self.t-1]
        self.trade_env["pProfit"][self.t] = 0

        # If no position exists
        risk_factor = 1.5 * self.calculate_atr()
        if self.trade_env["pType"][self.t] == 0:
            if action > 0.5:
                self.trade_env["pType"][self.t] = 1
                self.trade_env["pOpenPrice"][self.t] = self.trade_env["close"][self.t]
                self.trade_env["pSLPrice"][self.t] = self.trade_env["close"][self.t] - risk_factor
            elif action < -0.5:
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
            self.check_exit(action=action)

        # Placeholder: Update the state with actual logic based on the action
        self.state = self._get_observation()
        self.reward = self._calculate_reward(action)
        self.done = self.t >= len(self.trade_env) - 10 
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def render(self, mode='human'):
        print(f"State: {self.state}")

    # Special functions

    def check_exit(self, action):
        diff_pips = 0
        exit = False

        if self.trade_env["pType"][self.t] == 1:
            if action < -0.5:
                diff_pips = self.trade_env["close"][self.t-1] - self.trade_env["pOpenPrice"][self.t-1] * 10000
                exit = True
        else:
            if action > 0.5:
                diff_pips = self.trade_env["pOpenPrice"][self.t-1] - self.trade_env["close"][self.t-1] * 10000
                exit = True
        
        # Return if no exit signal was given
        if not exit:
            return
        
        # Add profit/loss to step
        diff = diff_pips * 10 # times pip value 
        self.trade_env["pProfit"][self.t] = diff - self.fees
        self.trade_env["eAccBalance"][self.t] += self.trade_env["pProfit"][self.t]
        if self.trade_env["pProfit"] > 0:
            self.trade_env["eWins"] += 1
        else:
            self.trade_env["eLosses"] += 1

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


# Example usage
if __name__ == "__main__":
    data = pd.DataFrame({
        'time': pd.date_range(start='1/1/2020', periods=100, freq='T'),
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100),
        'close': np.random.rand(100),
        'tick_volume': np.random.randint(1, 100, 100),
    })

    env = TradingEnv(batch_size=10)
    state = env.reset()
    print("Initial State:", state)

    for _ in range(10):
        action = env.action_space.sample()  # Sample random action
        state, reward, done, info = env.step(action[0])
        env.render()
        if done:
            break
