# trading_env.py
# --------------
# This module implements a custom OpenAI Gym environment (TradingEnv) for simulating trading activities.

import os
import gym
from gym import spaces
import numpy as np
import pandas as pd
import pandas_ta as ta  # For technical indicators

from config import (
    CSV_PATH,
    TRANSACTION_COST,
    NOISE_STD,
    MIN_EPISODE_LENGTH,
    MAX_TRADES_PER_DAY,
    MAX_HOLD_MINUTES,
    INITIAL_ACCOUNT_BALANCE,
    TRADE_PENALTY,
    MIN_HOLD_STEPS,
    MAX_TRADES_PER_EPISODE,
    BUY_PENALTY_MULTIPLIER,
    DEFAULT_SELL_BONUS_THRESHOLD,
    DEFAULT_SELL_BONUS_MULTIPLIER,
    DEFAULT_SELL_BONUS_THRESHOLD_TIER2,
    DEFAULT_SELL_BONUS_MULTIPLIER_TIER2,
    DEFAULT_SELL_BONUS_MULTIPLIER_TIER3,
    DEFAULT_SELL_BONUS_THRESHOLD_TIER3,
    DEFAULT_SELL_REWARD_SCALING,
    DEFAULT_HOLD_REWARD_SCALING
)

def load_clean_data(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    return df

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 df, 
                 transaction_cost=TRANSACTION_COST, 
                 noise_std=NOISE_STD, 
                 min_episode_length=MIN_EPISODE_LENGTH, 
                 max_trades_per_day=MAX_TRADES_PER_DAY,
                 max_hold_minutes=MAX_HOLD_MINUTES,
                 initial_balance=INITIAL_ACCOUNT_BALANCE,
                 trade_penalty=TRADE_PENALTY,
                 min_hold_steps=MIN_HOLD_STEPS,
                 max_trades_per_episode=MAX_TRADES_PER_EPISODE,
                 sell_bonus_threshold=DEFAULT_SELL_BONUS_THRESHOLD,
                 sell_bonus_multiplier=DEFAULT_SELL_BONUS_MULTIPLIER,
                 sell_bonus_threshold_tier2=DEFAULT_SELL_BONUS_THRESHOLD_TIER2,
                 sell_bonus_multiplier_tier2=DEFAULT_SELL_BONUS_MULTIPLIER_TIER2,
                 sell_bonus_multiplier_tier3=DEFAULT_SELL_BONUS_MULTIPLIER_TIER3,
                 sell_bonus_threshold_tier3=DEFAULT_SELL_BONUS_THRESHOLD_TIER3,
                 sell_reward_scaling=DEFAULT_SELL_REWARD_SCALING,
                 hold_reward_scaling=DEFAULT_HOLD_REWARD_SCALING):
        
        super(TradingEnv, self).__init__()
        self.df = df.copy().reset_index(drop=True)
        self.n_steps = len(self.df)
        self.transaction_cost = transaction_cost
        self.noise_std = noise_std
        self.min_episode_length = min_episode_length
        self.max_trades_per_day = max_trades_per_day
        self.max_hold_minutes = max_hold_minutes
        self.sell_reward_scaling = sell_reward_scaling
        self.hold_reward_scaling = hold_reward_scaling
        self.current_step = 0
        
        # Ensure required price columns exist
        if 'Open' not in self.df.columns:
            self.df['Open'] = self.df['Close']
        if 'High' not in self.df.columns:
            self.df['High'] = self.df['Close']
        if 'Low' not in self.df.columns:
            self.df['Low'] = self.df['Close']
        if 'Volume' not in self.df.columns:
            self.df['Volume'] = 1000

        # Calculate technical indicators using pandas_ta
        self.df.ta.sma(length=14, append=True)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.macd(append=True)
        self.df.ta.bbands(length=20, std=2, append=True)
        self.df.ta.atr(length=14, append=True)

        if 'ATRr_14' in self.df.columns:
            self.df.rename(columns={'ATRr_14': 'ATR_14'}, inplace=True)

        # Custom indicators
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Volatility'] = self.df['Returns'].rolling(window=14).std() * np.sqrt(14)
        self.df['AvgVolume'] = self.df['Volume'].rolling(window=14).mean()
        self.df['RelativeVolume'] = self.df['Volume'] / self.df['AvgVolume']

        self.df.fillna(0, inplace=True)

        # Define features for observations (14 features)
        self.feature_columns = [
            'Close', 'SMA_14', 'RSI_14', 'BBM_20_2.0', 'BBU_20_2.0',
            'BBL_20_2.0', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
            'ATR_14', 'Returns', 'Volatility', 'AvgVolume', 'RelativeVolume'
        ]
        missing_cols = set(self.feature_columns) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data after calculations: {missing_cols}")

        # Change action space to continuous for SAC:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),
            dtype=np.float32
        )

        # Internal state
        self.position = 0    # 0: flat, 1: long
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.trades = 0
        self.entry_time = None
        self.trades_today = 0
        self.current_date = None

        self.initial_balance = initial_balance
        self.account_balance = initial_balance

        self.trade_profits = []
        self.trade_dates = []
        self.trade_penalty = trade_penalty
        self.min_hold_steps = min_hold_steps
        self.max_trades_per_episode = max_trades_per_episode
        self.hold_steps = 0
        self.episode_trades = 0

        self.daily_balances = {}

    def _next_observation(self):
        obs = self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)
        noise = np.random.normal(0, self.noise_std, size=obs.shape)
        return obs + noise

    def calculate_reward(self, action, current_price, prev_price=None):
        if action == 1:  # Buy
            return -self.transaction_cost * BUY_PENALTY_MULTIPLIER - self.trade_penalty
        elif action == 2:  # Sell
            profit = current_price - self.entry_price
            net_profit = profit - self.transaction_cost
            reward = np.tanh(net_profit) * self.sell_reward_scaling
            reward -= self.trade_penalty
            return reward
        else:  # Hold
            if self.position == 1:
                incremental = current_price - (prev_price if prev_price else current_price)
                reward = np.tanh(incremental) * self.hold_reward_scaling
                return reward
            else:
                return 0.0

    def reset(self):
        self.current_step = np.random.randint(0, max(1, self.n_steps - self.min_episode_length))
        self.position = 0
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.trades = 0
        self.trades_today = 0
        self.current_date = self.df.loc[self.current_step, 'DateTime'].date()
        self.entry_time = None
        self.trade_profits = []
        self.trade_dates = []
        self.account_balance = self.initial_balance
        self.daily_balances = {}
        self.hold_steps = 0
        self.episode_trades = 0
        return self._next_observation()

    def step(self, action):
        # --- Convert continuous action to discrete ---
        # For SAC, action is a continuous value in [-1, 1]
        if isinstance(action, (np.ndarray, list)):
            action_val = action[0]
        else:
            action_val = action
        # Define thresholds to map continuous action to discrete decisions:
        if action_val > 0.33:
            action = 1   # Buy
        elif action_val < -0.33:
            action = 2   # Sell
        else:
            action = 0   # Hold
        # ---------------------------------------------

        current_price = self.df.loc[self.current_step, 'Close']
        prev_price = self.df.loc[self.current_step - 1, 'Close'] if self.current_step > 0 else current_price

        step_date = self.df.loc[self.current_step, 'DateTime'].date()
        if self.current_date is None or step_date != self.current_date:
            self.current_date = step_date
            self.trades_today = 0

        if self.position == 1:
            self.hold_steps += 1
        else:
            self.hold_steps = 0

        if self.position == 1 and self.hold_steps < self.min_hold_steps and action == 2:
            action = 0

        if self.episode_trades >= self.max_trades_per_episode:
            if self.position == 0 and action in [1, 2]:
                action = 0

        reward = 0.0
        if self.position == 0:
            if action == 1:  # Buy
                self.position = 1
                self.entry_price = current_price
                self.trades += 1
                self.trades_today += 1
                self.episode_trades += 1
                reward = self.calculate_reward(action, current_price) - self.trade_penalty
            else:
                reward = self.calculate_reward(0, current_price, prev_price)
        else:
            if action == 2:  # Sell
                reward = self.calculate_reward(action, current_price)
                profit = current_price - self.entry_price
                self.total_profit += profit
                self.trade_profits.append(profit)
                self.trade_dates.append(step_date)
                self.position = 0
                self.entry_price = 0.0
                self.trades += 1
                self.episode_trades += 1
                reward -= self.trade_penalty
            else:
                reward = self.calculate_reward(0, current_price, prev_price)

        self.account_balance += reward
        self.daily_balances[step_date] = self.account_balance

        self.current_step += 1
        done = (self.current_step >= self.n_steps - 1)

        if not done:
            obs = self._next_observation()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            'total_profit': self.total_profit,
            'trades': self.trades,
            'trade_profits': self.trade_profits,
            'daily_balances': self.daily_balances if done else None
        }
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}, "
              f"Position: {self.position}, "
              f"Account Balance: {self.account_balance:.2f}, "
              f"Total Profit (realized): {self.total_profit:.2f}, "
              f"Trades: {self.trades}")
