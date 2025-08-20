import numpy as np


class RewardFunctions:
    """
    Enumerated class of reward functions used for iteration to test reward returns of
    environments.
    """
    def __init__(self):
        self.functions = [
            self.reward_main,
            self.reward_normalized,
            self.reward_percentage_change,
            self.reward_profit_loss,
            self.reward_sharpe_ratio,
            self.reward_closing_high,
            self.reward_volume_weighted,
            self.reward_log_return,
            self.reward_binary_profit,
            self.reward_compound_growth,
            self.reward_environment_two,
            self.reward_simplistic_comp,
            self.reward_with_drawdown,
            self.reward_combined,
            self.reward_winrate,
            self.test_sellonly,
            self.reward_percentage_of_init,
            self.reward_growth_trajectory  # Add the new reward function here
        ]
        self.function_names = [func.__name__ for func in self.functions]
        self.func_name = "none"
        self.history_data = None

    def scale_reward(self, reward, min_val=-1, max_val=1):
        scaled_reward = (reward - min_val) / (max_val - min_val)
        return np.clip(scaled_reward, -1, 1)

    def reward_main(self, history, idx):
        """Raw percentage change in account balance between consecutive steps."""
        current_valuation = history["eAccBalance"][idx]
        previous_valuation = history["eAccBalance"][idx - 1]
        return (current_valuation - previous_valuation) / previous_valuation if previous_valuation != 0 else 0

    def reward_normalized(self, history, idx):
        """Percentage change in balance scaled to ``[-1, 1]`` using :func:`scale_reward`."""
        current_valuation = history["eAccBalance"][idx]
        previous_valuation = history["eAccBalance"][idx - 1]
        raw_reward = (current_valuation - previous_valuation) / previous_valuation if previous_valuation != 0 else 0
        normalized_reward = self.scale_reward(raw_reward, -1, 1)
        return normalized_reward

    def reward_percentage_change(self, history, idx):
        """
        Percentage Change in Portfolio Valuation - Reward based on the percentage change in portfolio valuation.
        """
        self.history_data = history
        previous_valuation = history['eAccBalance'][idx-1]
        current_valuation = history['eAccBalance'][idx]
        return (current_valuation - previous_valuation) / previous_valuation if previous_valuation != 0 else 0

    def reward_profit_loss(self, history, idx):
        """
        Profit and loss - Reward is the difference in portfolio valuation.
        """
        self.history_data = history
        return history['eAccBalance'][idx] - history['eAccBalance'][idx-1]

    def reward_sharpe_ratio(self, history, idx, risk_free_rate=0.01, epsilon=1e-8):
        returns = (history['eAccBalance'][1:idx + 1] - history['eAccBalance'][:idx]) / history['eAccBalance'][:idx]
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        if std_returns == 0:
            return 0

        sharpe_ratio = (mean_returns - risk_free_rate) / (std_returns + epsilon)
        return sharpe_ratio

    def reward_closing_high(self, history, idx):
        """
        Reward for closing high - Reward is the closing near the high of the day.
        """
        self.history_data = history
        close = history['close'][idx]
        high = history['high'][idx]
        low = history['low'][idx]
        return (close - high) / (high - low) if (high - low) != 0 else 0

    def reward_volume_weighted(self, history, idx):
        """
        Volume-weighted reward - Reward is based on volume and price movement.
        """
        self.history_data = history
        volume = history['tick_volume'][idx]
        price_movement = history['close'][idx] - history['open'][idx]
        return price_movement * volume

    def reward_log_return(self, history, idx):
        """
        Logarithmic return - Reward is the logarithmic return on portfolio valuation.
        """
        self.history_data = history
        return np.log(history['eAccBalance'][idx] / history['eAccBalance'][idx-1]) if history['eAccBalance'][idx-1] != 0 else 0

    def reward_binary_profit(self, history, idx):
        """
        Binary reward for profit - 1 for profit, 0 for loss.
        """
        self.history_data = history
        return 1 if history['eAccBalance'][idx] > history['eAccBalance'][idx-1] else 0

    def reward_compound_growth(self, history, idx):
        """
        Compound growth rate - Reward based on the compound growth rate of the portfolio.
        It assumes that the first recorded valuation can be accessed directly as the 'initial' valuation.
        """
        self.history_data = history
        initial_valuation = history['eAccBalance'][0]
        current_valuation = history['eAccBalance'][idx]

        if initial_valuation != 0 and idx > 0:
            return (current_valuation / initial_valuation) ** (1 / idx) - 1
        else:
            return 0  # Return 0 growth if initial valuation is zero or steps are not greater than 0.

    def reward_environment_two(self, history, idx):
        """
        Reward function based on the immediate financial result from trading decisions.
        It calculates the percentage change in total portfolio money from one step to the next.
        """
        self.history_data = history
        previous_total = history['eAccBalance'][idx-1]
        current_total = history['eAccBalance'][idx]
        return (current_total - previous_total) / previous_total if previous_total != 0 else 0

    def reward_simplistic_comp(self, history, idx):
        """Simple momentum reward comparing balance to eight steps prior."""
        self.history_data = history
        if idx < 10:
            return 0
        elif history['eAccBalance'][idx] > history['eAccBalance'][idx-8]:
            return 1
        else:
            return -1

    def reward_with_drawdown(self, history, idx):
        """Balance change penalized by drawdown from the running peak."""
        previous_balance = history['eAccBalance'][:idx].max()
        current_balance = history['eAccBalance'][idx]
        drawdown = (previous_balance - current_balance) / previous_balance
        return (current_balance - previous_balance) / previous_balance - drawdown

    def reward_combined(self, history, idx, risk_free_rate=0.01, epsilon=1e-8):
        """Combination of Sharpe ratio and max drawdown scaled into a single score."""
        # Calculate returns
        returns = (history['eAccBalance'][1:idx + 1] - history['eAccBalance'][:idx]) / history['eAccBalance'][:idx]

        # Calculate mean and standard deviation of returns
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)

        # Handle case where standard deviation is zero
        if std_returns == 0:
            return 0  # Return zero if no variability

        # Calculate Sharpe ratio
        sharpe_ratio = (mean_returns - risk_free_rate) / (std_returns + epsilon)

        # Calculate max drawdown
        peak = np.maximum.accumulate(history['eAccBalance'][:idx])
        drawdown = (peak - history['eAccBalance'][:idx]) / peak
        max_drawdown = np.max(drawdown)

        # Calculate combined reward
        raw_reward = (sharpe_ratio - max_drawdown) * 100  # Scaling factor for reward

        # Normalize reward to ensure non-negative values
        normalized_reward = raw_reward - min(raw_reward, 0)  # Ensure the reward is at least zero

        return normalized_reward

    def reward_winrate(self, history, idx, win_rate=0.65):
        """Difference between current win rate and a target ``win_rate``."""
        wins = history["eWins"][idx]
        losses = history["eLosses"][idx]
        return ((wins / (wins + losses)) - win_rate) if wins + losses > 0 else 1

    def test_sellonly(self, history, idx):
        """Toy reward that favors sell actions and penalizes buys."""
        if history["pType"][idx] == 2:
            return 1
        elif history["pType"][idx] == 1:
            return -1
        else:
            return 0

    def reward_percentage_of_init(self, history, idx):
        """Percentage change in balance relative to the initial valuation."""
        current_valuation = history["eAccBalance"][idx]
        initial_valuation = history["eAccBalance"][0]
        return (current_valuation - initial_valuation) / initial_valuation

    def reward_growth_trajectory(self, history, idx, window=10):
        """
        Reward based on the moving average of the portfolio valuation growth.
        The agent is rewarded if the moving average of the derivative is positive.
        """
        if idx < window:
            return 0  # Not enough data to compute the moving average

        # Compute the moving average of the portfolio valuation changes
        portfolio_valuations = history["eAccBalance"][idx-window:idx]
        valuation_changes = np.diff(portfolio_valuations)
        moving_average_change = np.mean(valuation_changes)

        # Reward is the moving average change (scaled to ensure it's within a reasonable range)
        scaled_reward = self.scale_reward(moving_average_change, -1, 1)
        return scaled_reward

    def __getitem__(self, index):
        self.func_name = self.function_names[index]
        return self.functions[index]

    def __len__(self):
        return len(self.functions)
