# symbol_monitor.py

import asyncio
import logging
import math
from ib_insync import Stock, LimitOrder, MarketOrder, Ticker
from utils import round_to_tick
from config import (
    INITIAL_STOP_LOSS_PERCENTAGE,
    TRAILING_STOP_LOSS_PERCENTAGE,
    STOP_LOSS_FIXED_TRIGGER,
    PRE_MARKET_BUFFER,
    LOG_LEVEL,
    MARKET_DATA_TYPE
)
import pytz
from datetime import datetime, time as dt_time

class SymbolMonitor:
    def __init__(self, symbol, total_quantity, entry_price, tick_size, starting_ema_values, ibkr_client):
        self.symbol = symbol.upper()  # Ensure symbol is uppercase
        self.total_quantity = total_quantity  # Initial quantity
        self.entry_price = entry_price
        self.tick_size = tick_size
        self.starting_ema_values = starting_ema_values
        self.contract = Stock(self.symbol, "SMART", "USD")
        self.ib = ibkr_client.ib
        self.ibkr_client = ibkr_client
        self.ticker = None  # Will hold the Ticker object for this symbol
        self.highest_price = entry_price
        # Set initial stop loss
        self.stop_loss_price = round_to_tick(entry_price * (1 - INITIAL_STOP_LOSS_PERCENTAGE), tick_size)
        self.trailing = True
        # Define quantities for multi-step take profit
        self.quantities = {
            'first_sell': int(total_quantity * 0.25),
            'second_sell': int(total_quantity * 0.25),
            'final_sell': total_quantity - (int(total_quantity * 0.25) + int(total_quantity * 0.25))
        }
        self.first_sell_done = False
        self.second_sell_done = False
        # Initialize EMA values
        self.ema3, self.ema9, self.ema12 = starting_ema_values if starting_ema_values else (None, None, None)
        # Parameters for dynamic adjustments
        self.take_profit_multiplier = 0.05  # 5% take profit
        self.stop_loss_multiplier = 0.02    # 2% stop loss
        # Stop loss fixed trigger
        self.stop_loss_fixed_trigger_price = self.entry_price * (1 + STOP_LOSS_FIXED_TRIGGER)
        self.stop_loss_fixed = False  # Flag to indicate if stop loss is fixed
        # Event to cancel monitoring loop
        self.monitoring = True

    async def monitor_order(self, trade):
        try:
            # Subscribe to error events
            logging.info(f"[{self.symbol}] Starting to monitor order {trade.order.orderId}.")
            self.ib.errorEvent += self.on_error
            while not trade.isDone():
                await asyncio.sleep(1)
            logging.info(f"[{self.symbol}] Order {trade.order.orderId} completed with status {trade.orderStatus.status}")
            if trade.orderStatus.status == 'Filled' and trade.order.action == 'BUY':
                # Update entry price to actual fill price if available
                actual_fill_price = getattr(trade.orderStatus, 'avgFillPrice', None)
                if actual_fill_price and actual_fill_price > 0 and actual_fill_price != self.entry_price:
                    self.entry_price = actual_fill_price
                    self.stop_loss_price = round_to_tick(self.entry_price * (1 - self.stop_loss_multiplier), self.tick_size)
                    self.stop_loss_fixed_trigger_price = self.entry_price * (1 + STOP_LOSS_FIXED_TRIGGER)
                    logging.info(f"[{self.symbol}] Entry price updated to {self.entry_price:.6f}, new stop loss set at {self.stop_loss_price:.6f}")
                # Set profit targets
                self.set_take_profit_levels()
                await self.monitor_price()
        except Exception as e:
            logging.error(f"[{self.symbol}] Error monitoring order {trade.order.orderId}: {e}", exc_info=True)

    def on_error(self, reqId, errorCode, errorString, contract):
        if contract and contract.symbol.upper() == self.symbol and errorCode in (10168, 10089):
            logging.error(f"[{self.symbol}] Market data error for {self.symbol}. Error {errorCode}: {errorString}")
            # Initiate closing the position and possibly add symbol to blacklist
            asyncio.create_task(self.handle_market_data_error())

    async def handle_market_data_error(self):
        # Close the position
        await self.ibkr_client.ensure_connection()
        positions = await self.ib.reqPositionsAsync()

        # Filter positions for the current symbol
        symbol_positions = [
            pos for pos in positions
            if pos.contract.symbol.upper() == self.symbol
        ]

        # Sum the positions to get total quantity
        symbol_position = sum(pos.position for pos in symbol_positions)

        if symbol_position > 0:
            sell_qty = symbol_position  # Sell remaining shares
            logging.info(f"[{self.symbol}] Closing position due to market data error. Selling {sell_qty} shares.")
            await self.place_sell_order(sell_qty, self.entry_price)  # Use entry price or a fallback
            
            # Update the global tracker to reflect that the position is closed
            from order_management import open_positions
            open_positions[self.symbol] = 0

        # Stop monitoring
        self.monitoring = False


    def set_take_profit_levels(self):
        # Dynamically adjust profit targets
        self.first_profit_target = self.entry_price + (self.entry_price * self.take_profit_multiplier)
        self.second_profit_target = self.entry_price + (self.entry_price * self.take_profit_multiplier * 1.5)
        logging.info(f"[{self.symbol}] Profit targets set: First at {self.first_profit_target:.6f}, Second at {self.second_profit_target:.6f}")

    async def monitor_price(self):
        try:
            await self.ibkr_client.ensure_connection()
            self.ib.reqMarketDataType(MARKET_DATA_TYPE)  # Live data
            # Request market data for this symbol
            self.ticker = self.ib.reqMktData(self.contract, '', False, False)
            logging.info(f"[{self.symbol}] Monitoring started.")

            while self.monitoring:
                await asyncio.sleep(1)

                # Ensure the ticker has updated data
                if self.ticker.last is None or math.isnan(self.ticker.last):
                    logging.debug(f"[{self.symbol}] Current price is not available. Skipping iteration.")
                    continue

                current_price = float(self.ticker.last)

                # Request updated positions
                positions = await self.ib.reqPositionsAsync()

                # Filter positions for the current symbol (case-insensitive)
                symbol_positions = [
                    pos for pos in positions
                    if pos.contract.symbol.upper() == self.symbol
                ]

                # Sum the positions to get total quantity
                symbol_position = sum(pos.position for pos in symbol_positions)

                if symbol_position == 0:
                    logging.info(f"[{self.symbol}] Position closed. Stopping monitoring.")
                    break

                # Update EMA values
                smoothing3 = 2 / (3 + 1)
                smoothing9 = 2 / (9 + 1)
                smoothing12 = 2 / (12 + 1)

                if self.ema3 is not None:
                    self.ema3 = (current_price - self.ema3) * smoothing3 + self.ema3
                else:
                    self.ema3 = current_price

                if self.ema9 is not None:
                    self.ema9 = (current_price - self.ema9) * smoothing9 + self.ema9
                else:
                    self.ema9 = current_price

                if self.ema12 is not None:
                    self.ema12 = (current_price - self.ema12) * smoothing12 + self.ema12
                else:
                    self.ema12 = current_price

                # Log current price and EMAs
                logging.info(
                    f"[{self.symbol}] Price: {current_price:.6f} | EMA3: {self.ema3:.6f} | EMA9: {self.ema9:.6f} | EMA12: {self.ema12:.6f}")

                # Check if the stop loss should be fixed
                if not self.stop_loss_fixed and current_price >= self.stop_loss_fixed_trigger_price:
                    self.stop_loss_fixed = True
                    old_stop_loss = self.stop_loss_price
                    self.stop_loss_price = round_to_tick(
                        current_price * (1 - PRE_MARKET_BUFFER),
                        self.tick_size)
                    logging.info(
                        f"[{self.symbol}] Stop loss fixed. Price reached {current_price:.6f} (trigger: {self.stop_loss_fixed_trigger_price:.6f}). "
                        f"Stop loss updated from {old_stop_loss:.6f} to {self.stop_loss_price:.6f}.")

                # Update highest price and trailing stop loss if not fixed
                if not self.stop_loss_fixed and current_price > self.highest_price:
                    old_high = self.highest_price
                    old_stop_loss = self.stop_loss_price
                    self.highest_price = current_price
                    self.stop_loss_price = round_to_tick(
                        self.highest_price * (1 - (TRAILING_STOP_LOSS_PERCENTAGE * 1.5)),
                        self.tick_size)
                    logging.debug(
                        f"[{self.symbol}] New high updated from {old_high:.6f} to {self.highest_price:.6f}. "
                        f"Trailing stop loss adjusted from {old_stop_loss:.6f} to {self.stop_loss_price:.6f}.")

                # Check if current price has fallen below stop loss
                if current_price <= self.stop_loss_price:
                    sell_qty = symbol_position  # Sell remaining shares
                    logging.info(f"[{self.symbol}] Stop loss hit at {current_price:.6f}. Selling {sell_qty} shares.")
                    await self.place_sell_order(sell_qty, current_price)
                    from order_management import open_positions
                    open_positions[self.symbol] = 0
                    break  # Exit after placing sell order

                # Multi-step take profit logic
                if not self.first_sell_done and current_price >= self.first_profit_target:
                    self.first_sell_done = True
                    sell_qty = self.quantities['first_sell']
                    logging.info(f"[{self.symbol}] First profit target reached at {current_price:.6f}. Selling {sell_qty} shares.")
                    await self.place_sell_order(sell_qty, current_price)
                    from order_management import open_positions
                    open_positions[self.symbol] = max(open_positions.get(self.symbol, 0) - sell_qty, 0)
                    # Update stop loss to just below the current price
                    self.stop_loss_price = round_to_tick(current_price * (1 - PRE_MARKET_BUFFER), self.tick_size)
                    logging.info(f"[{self.symbol}] Stop loss updated to {self.stop_loss_price:.6f} after first sell.")
                elif self.first_sell_done and not self.second_sell_done and current_price >= self.second_profit_target:
                    self.second_sell_done = True
                    sell_qty = self.quantities['second_sell']
                    logging.info(f"[{self.symbol}] Second profit target reached at {current_price:.6f}. Selling {sell_qty} shares.")
                    await self.place_sell_order(sell_qty, current_price)
                    from order_management import open_positions
                    open_positions[self.symbol] = max(open_positions.get(self.symbol, 0) - sell_qty, 0)
                    # Update stop loss to just below the current price
                    self.stop_loss_price = round_to_tick(current_price * (1 - PRE_MARKET_BUFFER), self.tick_size)
                    logging.info(f"[{self.symbol}] Stop loss updated to {self.stop_loss_price:.6f} after second sell.")

        except Exception as e:
            logging.error(f"[{self.symbol}] Error monitoring price: {e}", exc_info=True)
        finally:
            # Cancel market data subscription when done
            if self.ticker:
                self.ib.cancelMktData(self.contract)
            # Unsubscribe from error events
            self.ib.errorEvent -= self.on_error

    async def place_sell_order(self, quantity, current_price):
        try:
            await self.ibkr_client.ensure_connection()
            if quantity <= 0:
                logging.warning(f"[{self.symbol}] No quantity left to sell.")
                return

            # Define market hours (PST)
            market_open = dt_time(6, 30)  # 6:30 AM
            market_close = dt_time(13, 0)  # 1:00 PM
            pst_timezone = pytz.timezone('US/Pacific')
            now_pst = datetime.now(pst_timezone).time()

            if market_open <= now_pst <= market_close:
                # Place a market order during market hours
                sell_order = MarketOrder("SELL", quantity, outsideRth=False)
                order_type = 'Market'
            else:
                # Place a limit order outside market hours
                buffer = PRE_MARKET_BUFFER
                limit_price = round_to_tick(current_price * (1 - buffer), self.tick_size)
                sell_order = LimitOrder("SELL", quantity, limit_price, outsideRth=True, tif='DAY')
                order_type = f'Limit at ${limit_price:.6f}'

            sell_trade = self.ib.placeOrder(self.contract, sell_order)
            await self.ibkr_client.get_account_balance()
            logging.info(f"[{self.symbol}] Placed {order_type} SELL order for {quantity} shares.")
            # Log the trade with extra info indicating the source.
            from trade_logger import log_trade
            log_trade(self.symbol, "SELL", quantity, current_price, sell_trade.order.orderId, extra_info=f"{order_type} from SymbolMonitor.place_sell_order")
            asyncio.create_task(self.monitor_sell_order(sell_trade))
        except Exception as e:
            logging.error(f"[{self.symbol}] Error placing sell order: {e}", exc_info=True)

    async def monitor_sell_order(self, trade):
        try:
            while not trade.isDone():
                await asyncio.sleep(1)
            logging.info(f"[{self.symbol}] Sell order {trade.order.orderId} completed with status {trade.orderStatus.status}")
        except Exception as e:
            logging.error(f"[{self.symbol}] Error monitoring sell order {trade.order.orderId}: {e}", exc_info=True)
