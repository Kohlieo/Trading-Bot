
# main.py
# -------
# This file serves as the entry point for the trading bot application. It performs the following functions:
#   - Configures logging using settings from config.py
#   - Initiates and establishes the IBKR connection (via ibkr_client)
#   - Retrieves account balance and synchronizes positions upon startup
#   - Sets up signal handlers to gracefully shutdown all asynchronous tasks
#   - Starts the RL trading process by instantiating and running the RLTrader class
#   - Launches a web server for webhook/API handling
# Overall, this file orchestrates the initialization, supervision, and orderly shutdown of the trading bot.

from datetime import datetime
import asyncio
import logging
import signal
from ibkr_connection import ibkr_client
from webhook_handler import app
from config import LOG_LEVEL
from rl_trader import RLTrader

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def shutdown(loop, signal_name=None):
    logging.info(f"[{datetime.now()}] Received exit signal {signal_name}. Shutting down gracefully...")
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    logging.debug(f"[{datetime.now()}] Shutting down {len(tasks)} pending tasks.")
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await ibkr_client.disconnect()
    loop.stop()
    logging.debug(f"[{datetime.now()}] Shutdown complete, event loop stopped.")

def setup_signal_handlers(loop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda sig=sig: asyncio.create_task(shutdown(loop, sig.name)))

async def supervisor():
    loop = asyncio.get_running_loop()
    setup_signal_handlers(loop)
    try:
        logging.debug(f"[{datetime.now()}] Starting IBKR connection process...")
        await ibkr_client.connect()
        await ibkr_client.get_account_balance()
        await ibkr_client.sync_positions_with_ibkr()
        logging.debug(f"[{datetime.now()}] IBKR connection and initialization complete.")
    except Exception as e:
        logging.error(f"[{datetime.now()}] Error during IBKR initialization: {e}", exc_info=True)
        # Instead of calling shutdown() here, we log the error and continue.
        # This ensures shutdown only occurs if ctrl+c is pressed.
    
    trader = RLTrader()
    trader_task = asyncio.create_task(trader.run())
    web_server_task = asyncio.create_task(app.run_task(debug=False, host='0.0.0.0', port=5000))
    
    # Instead of waiting for FIRST_EXCEPTION, wait indefinitely.
    try:
        await asyncio.gather(trader_task, web_server_task)
    except Exception as e:
        logging.error(f"[{datetime.now()}] An unexpected error occurred: {e}", exc_info=True)
        # Optionally, decide whether to continue running or exit.
        # Here we simply log the error.
    
    # If either task completes normally, we exit the supervisor.
    await ibkr_client.disconnect()
    logging.debug(f"[{datetime.now()}] Supervisor completed.")

if __name__ == "__main__":
    try:
        asyncio.run(supervisor())
    except Exception as e:
        logging.error(f"[{datetime.now()}] Supervisor encountered an error: {e}", exc_info=True)
