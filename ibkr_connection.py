# ibkr_connection.py
# -------------------
# This module implements the IBKRClient class to manage connectivity with Interactive Brokers (IBKR). Key functions include:
#   - Establishing and ensuring a stable asynchronous connection to IBKR (connect, ensure_connection)
#   - Disconnecting safely from IBKR
#   - Retrieving account balance and synchronizing positions with IBKR
#   - Determining appropriate tick sizes for given symbols based on current market prices
#   - Cancelling active orders when needed
# The module leverages the ib_insync library and patches the asyncio event loop to seamlessly integrate IBKR API operations.

import asyncio
import logging
from ib_insync import IB, util, Stock
from config import IB_HOST, IB_PORT, IB_CLIENT_ID, TICK_SIZE_FILE, MARKET_DATA_TYPE
from utils import load_tick_sizes, save_tick_sizes
import math

# Patch asyncio event loop for ib_insync
util.patchAsyncio()

class IBKRClient:
    def __init__(self):
        self.ib = IB()
        self.available_equity = None
        self.state_lock = asyncio.Lock()
        self.tick_size_cache = load_tick_sizes()
    
    async def connect(self):
        if not self.ib.isConnected():
            logging.debug(f"[{self._current_time()}] Initiating connection to IBKR...")
            logging.info(f"[{self._current_time()}] Connecting to {IB_HOST}:{IB_PORT} with client ID {IB_CLIENT_ID}")
            try:
                await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
                if self.ib.isConnected():
                    logging.info(f"[{self._current_time()}] Successfully connected to IBKR.")
                else:
                    logging.error(f"[{self._current_time()}] Connection attempt finished but IB instance is not connected.")
            except Exception as e:
                logging.error(f"[{self._current_time()}] Failed to connect to IBKR: {e}", exc_info=True)
                raise e

    async def disconnect(self):
        logging.debug(f"[{self._current_time()}] Disconnect requested. Checking connection status before disconnecting.")
        if self.ib.isConnected():
            self.ib.disconnect()
            logging.info(f"[{self._current_time()}] Disconnected from IBKR.")
        else:
            logging.info(f"[{self._current_time()}] Disconnect requested, but already disconnected.")

    async def ensure_connection(self, retries=3, delay=5):
        attempt = 0
        while not self.ib.isConnected() and attempt < retries:
            attempt += 1
            logging.warning(f"[{self._current_time()}] IBKR disconnected. Attempting to reconnect (attempt {attempt}/{retries})...")
            try:
                await self.connect()
            except Exception as e:
                logging.error(f"[{self._current_time()}] Reconnect attempt {attempt} failed: {e}", exc_info=True)
            if not self.ib.isConnected():
                logging.debug(f"[{self._current_time()}] Waiting for {delay} seconds before next connection attempt...")
                await asyncio.sleep(delay)
        if not self.ib.isConnected():
            error_message = f"[{self._current_time()}] Failed to reconnect to IBKR after {retries} attempts"
            logging.error(error_message)
            raise ConnectionError(error_message)
        logging.debug(f"[{self._current_time()}] ensure_connection: IBKR is connected.")
        return True

    def _current_time(self):
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    async def get_account_balance(self):
        try:
            await self.ensure_connection()
            account_summary = await self.ib.accountSummaryAsync()
            for account_value in account_summary:
                if account_value.tag == 'AvailableFunds':
                    if account_value.currency == 'CAD':
                        async with self.state_lock:
                            self.available_equity = float(account_value.value)
                            logging.info(f"Available Equity (CAD): {self.available_equity}")
                        break
                    elif self.available_equity is None:
                        async with self.state_lock:
                            self.available_equity = float(account_value.value)
                            logging.info(f"Available Equity ({account_value.currency}): {self.available_equity}")
            if self.available_equity is None:
                logging.error("Available equity not found in account summary.")
        except Exception as e:
            logging.error(f"Error getting account balance: {e}", exc_info=True)

    async def sync_positions_with_ibkr(self):
        try:
            await self.ensure_connection()
            positions = await asyncio.to_thread(self.ib.positions)
            for position in positions:
                symbol = position.contract.symbol
                logging.info(f"Position: {symbol}, Quantity: {position.position}, Avg Cost: {position.avgCost}")
        except Exception as e:
            logging.error(f"Error syncing positions with IBKR: {e}", exc_info=True)

    async def get_tick_size(self, symbol):
        """
        Determine the tick size based on the stock's current price.
        For stocks priced at $1 or above, use 0.01.
        For stocks below $1, use 0.0001.
        This function fetches the current price and then applies these defaults.
        """
        # Qualify contract and set market data type
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        self.ib.reqMarketDataType(MARKET_DATA_TYPE)
        ticker = self.ib.reqMktData(contract, '', False, False)
        
        # Wait (up to 5 seconds) for a valid price
        total_wait = 0.0
        interval = 0.1
        while ticker.last is None or (isinstance(ticker.last, float) and math.isnan(ticker.last)):
            await asyncio.sleep(interval)
            total_wait += interval
            if total_wait > 5:
                break

        # Use the fetched price or default to $1.0 if unavailable
        price = ticker.last if ticker.last is not None and not math.isnan(ticker.last) else 1.0

        # Determine tick size based on price threshold
        if price >= 1:
            tick_size = 0.01
        else:
            tick_size = 0.0001

        logging.info(f"Determined tick size for {symbol} based on price {price}: {tick_size}")
        return tick_size


    async def cancel_active_orders(self, symbol):
        try:
            await self.ensure_connection()
            for trade in self.ib.trades():
                order = trade.order
                if trade.contract.symbol == symbol and not trade.isDone():
                    self.ib.cancelOrder(order)
                    while trade.orderStatus.status not in ('Cancelled', 'Filled'):
                        await asyncio.sleep(0.1)
                    logging.info(f"Cancelled order {order.orderId} for {symbol}.")
        except Exception as e:
            logging.error(f"Error cancelling orders for {symbol}: {e}", exc_info=True)

# Instantiate ibkr_client at the module level
ibkr_client = IBKRClient()
