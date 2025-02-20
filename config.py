# config.py

import os

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# IBKR connection settings
IB_HOST = os.getenv('IB_HOST', 'host.docker.internal')  # Adjust as needed
IB_PORT = int(os.getenv('IB_PORT', 7496))
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', 123))

# Trailing stop parameters
TAKE_PROFIT_PERCENTAGE = 0.05  # 5% gain
INITIAL_STOP_LOSS_PERCENTAGE = 0.02  # 2% initial stop loss
TRAILING_STOP_LOSS_PERCENTAGE = 0.03  # 3% trailing stop
STOP_LOSS_FIXED_TRIGGER = 0.02  # 2% above entry price where stop loss becomes fixed
PRE_MARKET_BUFFER = 0.005  # 0.5% buffer for pre-market volatility

# Tick size cache file path
TICK_SIZE_FILE = 'tick_sizes.json'  # Changed to JSON format

# Available equity percentage to use per trade
TRADE_EQUITY_PERCENTAGE = 0.65  # 65% of available equity

# RL settings (assuming these were previously added)
RL_MODEL_PATH = os.getenv('RL_MODEL_PATH', 'latest_trading_bot_model')
RL_SYMBOLS = ["STAI", "EFOI", "SLDB", "PRFX"]
MAX_SYMBOLS = 5

MARKET_DATA_TYPE = int(os.getenv('MARKET_DATA_TYPE', 3))  # 1 for live data, 3 for delayed data

# ---------------------------
# Data and File Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "clean_price_data.csv")
MODEL_NAME = "latest_trading_bot_model"

# ---------------------------
# Trading Environment Parameters
# ---------------------------
TRANSACTION_COST = 0.001
NOISE_STD = 0.001
MIN_EPISODE_LENGTH = 300
MAX_TRADES_PER_DAY = 10
MAX_HOLD_MINUTES = 80

# Penalty / Reward Multipliers
BUY_PENALTY_MULTIPLIER = 4
TRADE_PENALTY = 0.5
MIN_HOLD_STEPS = 4
MAX_TRADES_PER_EPISODE = 20

# Reward scaling parameters (default values)

DEFAULT_SELL_BONUS_MULTIPLIER = 108.68
DEFAULT_SELL_BONUS_THRESHOLD = 1.89
DEFAULT_SELL_BONUS_MULTIPLIER_TIER2 = 231.45
DEFAULT_SELL_BONUS_THRESHOLD_TIER2 = 2.4
DEFAULT_SELL_BONUS_MULTIPLIER_TIER3 = 721.01
DEFAULT_SELL_BONUS_THRESHOLD_TIER3 = 5.06
DEFAULT_SELL_REWARD_SCALING = 199.64
DEFAULT_HOLD_REWARD_SCALING = 11.59

# Initial account balance for simulation
INITIAL_ACCOUNT_BALANCE = 1000.0

# ---------------------------
# RL Training Parameters
# ---------------------------
TOTAL_TIMESTEPS = 100_000
TUNING_TRIALS = 10
TUNING_TOTAL_TIMESTEPS = 10_000
TUNING_N_EPISODES_EVALUATION = 2
N_EPISODES_EVALUATION = 10
