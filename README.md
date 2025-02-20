# Trading Bot
 My ibkr trading bot

Trading Bot README

This trading bot is a modular, asynchronous trading system designed to connect with Interactive Brokers (IBKR) and execute trades using a reinforcement learning (RL) strategy. It integrates real-time market data, order management, dynamic position monitoring, and a custom trading environment for offline training and simulation. The bot is built in Python and leverages several key libraries including ib_insync for IBKR connectivity, stable-baselines3 (PPO) for RL, gym for the simulation environment, and pandas_ta for technical indicators.
Overview

    IBKR Connectivity:
    The bot connects to Interactive Brokers via an asynchronous IBKR client that manages connections, fetches account details, retrieves market data, and handles order cancellations. The connection parameters (host, port, client ID, etc.) and trading parameters (stop loss, take profit, tick sizes, etc.) are configured in a dedicated configuration file.

,

Order Management:
Orders (both BUY and SELL) are placed via a dedicated module that:

    Checks account equity and existing positions.
    Determines the proper tick size for pricing.
    Adjusts order prices using a pre-market buffer.
    Logs each trade in a CSV file for audit and analysis. The module also manages closing all open positions and monitors active orders.

,

Reinforcement Learning Trader:
The RLTrader class implements an RL strategy using a pre-trained PPO model:

    It subscribes to real-time market data for a list of symbols.
    Price ticks are buffered and technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR, etc.) are computed using pandas_ta.
    Based on computed features, the RL model predicts whether to BUY, SELL, or HOLD.
    The bot enforces trade cycle limits and ensures that trades occur only during defined trading hours.
    It also supports online incremental training using data collected during the trading day.

Symbol Monitoring:
After an order is executed, the SymbolMonitor class tracks the symbol’s price:

    It continuously updates the highest price and adjusts trailing stop losses.
    Implements multi-step take profit logic by partially liquidating positions at predefined profit targets.
    Uses exponential moving averages (EMA) for dynamic decision making and can fix stop loss levels when a certain threshold is reached.
    In case of market data errors, it automatically initiates position closures. 

Trading Environment (Gym):
A custom OpenAI Gym environment simulates trading using historical price data:

    It computes various technical indicators and builds a 14-feature observation space.
    The environment supports discrete actions (0=Hold, 1=Buy, 2=Sell) and calculates rewards based on transaction costs, penalties, and profit scaling.
    This simulation environment is used for both offline training and online incremental updates. 

Utilities and Trade Logging:
Helper functions are provided to:

    Load and save tick size information from a persistent JSON file.
    Round values to the nearest tick. Trade details (timestamp, symbol, action, quantity, price, order ID, and additional info) are recorded in a CSV file for later review.

,

Application Entry Point:
The main.py file initializes the system:

    It sets up the logging configuration.
    Connects to IBKR and fetches account balances.
    Starts the RLTrader’s asynchronous trading loop alongside a web server (likely for webhook/API handling).
    Gracefully handles shutdown signals ensuring all tasks and connections are closed properly. 

Configuration

The configuration parameters (located in config.py) allow you to adjust:

    Logging level: Control verbosity.
    IBKR connection details: Host, port, and client ID.
    Trading Parameters:
        Trailing and initial stop-loss percentages.
        Take profit thresholds.
        Pre-market buffers for price adjustments.
        Equity allocation per trade.
    Reinforcement Learning Settings:
        Model path and list of trading symbols.
        Total timesteps for training and online training parameters.
    Simulation Environment Parameters:
        Transaction costs, noise levels, minimum episode lengths, etc.

This file centralizes key settings to easily tune the bot’s performance.
Dependencies

    Python 3.7+
    ib_insync: For IBKR API connectivity and asynchronous operations.
    stable-baselines3: Implements the PPO RL model.
    gym: For creating the trading simulation environment.
    pandas & pandas_ta: For data manipulation and technical indicator calculations.
    asyncio: For asynchronous task management.
    Other Libraries: Standard libraries such as logging, datetime, math, and csv are used throughout.

Installation & Setup

    Clone the Repository:

git clone <repository-url>
cd trading-bot

Install Dependencies: Ensure you have Python 3.7 or higher, then install required packages:

pip install ib_insync stable-baselines3 gym pandas pandas_ta asyncio

Configure Environment Variables:
You can customize settings via environment variables (e.g., IB_HOST, IB_PORT, LOG_LEVEL, etc.) or by modifying config.py.

Prepare Shared Directories:
Ensure the shared directory (used by the trade logger and ticker files) exists or adjust paths in the configuration.

Run the Bot: Start the application with:

    python main.py

Usage

    Live Trading:
    The bot connects to IBKR, subscribes to market data, and automatically makes trade decisions based on RL model outputs. It enforces trading windows and trade cycle limits to ensure risk management.

    Offline Training:
    Use the custom Gym environment (in trading_env.py) to simulate trading and further train the RL model using historical price data.

    Monitoring and Logging:
    All executed trades are logged to a CSV file (trade_log.csv), and detailed logs are available based on the set logging level. This helps in post-trade analysis and debugging.

    Error Handling:
    The system continuously monitors for connection issues or market data errors and will attempt to reconnect or close positions when necessary.

Extending the Bot

    Strategy Adjustments:
    Modify the RL model or adjust the trading signals by tuning the technical indicator computations in RLTrader or updating the decision logic in the custom Gym environment.

    Order Management Enhancements:
    Additional order types or more granular risk management features can be added within the order management module.

    Integration with Webhooks/APIs:
    The web server started in main.py can be expanded to expose API endpoints for monitoring, manual trade interventions, or dashboard integrations.

This trading bot offers a robust framework for automated trading with a blend of real-time execution and reinforcement learning. Each module is designed to be modular and extensible, ensuring that traders and developers can fine-tune the system according to evolving market strategies and risk management requirements.

By understanding each component and how they interact, you can confidently deploy, monitor, and further develop this trading system.

Happy Trading!