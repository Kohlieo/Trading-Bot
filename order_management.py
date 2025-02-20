
# order_management.py
# -------------------
# This module handles the placement and management of trade orders. Its responsibilities include:
#   - Placing BUY and SELL orders with proper price adjustments (using market data and a pre-market buffer)
#   - Checking for sufficient equity and verifying open positions before executing orders
#   - Rounding prices to valid tick sizes for order accuracy
#   - Logging trade details via the trade_logger module
#   - Monitoring orders until they are filled or cancelled
#   - Providing a function to close all open positions for a given symbol
# This module integrates closely with the IBKR client and the symbol monitoring system to ensure reliable order execution.

import asyncio
import logging
import math
from ib_insync import Stock, LimitOrder, MarketOrder
from ibkr_connection import ibkr_client
from config import PRE_MARKET_BUFFER, TRADE_EQUITY_PERCENTAGE, MARKET_DATA_TYPE
from symbol_monitor import SymbolMonitor
from utils import round_to_tick
from trade_logger import log_trade  # Import the trade logger

# Global internal position tracker: maps symbol (uppercase) to quantity.
open_positions = {}

async def place_order(symbol, action, price=None, starting_ema_values=None):
    ib = ibkr_client.ib  # Access the IB instance
    symbol = symbol.upper()
    try:
        await ibkr_client.ensure_connection()
        tick_size = await ibkr_client.get_tick_size(symbol)
        logging.info(f"Tick size for {symbol}: {tick_size}")

        # Set market data type using the integer from config
        ib.reqMarketDataType(MARKET_DATA_TYPE)
        buffer = PRE_MARKET_BUFFER

        if action.upper() == "SELL":
            # Use internal tracker to verify an open position exists.
            if open_positions.get(symbol, 0) <= 0:
                logging.info(f"No open position in {symbol} to sell. Skipping SELL.")
                return {"status": f"No position in {symbol} to sell."}
            current_position = open_positions[symbol]

            # If no price provided, fetch one.
            if price is None:
                contract = Stock(symbol, "SMART", "USD")
                ticker = ib.reqMktData(contract, '', False, False)
                total_wait = 0.0
                while ticker.last is None or (isinstance(ticker.last, float) and math.isnan(ticker.last)):
                    await asyncio.sleep(0.1)
                    total_wait += 0.1
                    if total_wait > 5:
                        logging.error(f"Fetched latest price for {symbol} is still NaN after waiting.")
                        return {"error": "Fetched latest price is NaN"}
                price = ticker.last
                logging.info(f"Fetched latest price for {symbol}: {price}")

            if math.isnan(price):
                logging.error(f"Latest price for {symbol} is NaN. Aborting order placement.")
                return {"error": "Fetched latest price is NaN"}

            adjusted_price = round_to_tick(price * (1 - buffer), tick_size)
            if math.isnan(adjusted_price):
                logging.error(f"Computed adjusted price for {symbol} is NaN. Aborting order placement.")
                return {"error": "Adjusted price is NaN"}

            # Place a limit SELL order to close the position.
            contract = Stock(symbol, "SMART", "USD")
            main_order = LimitOrder(
                "SELL",
                current_position,
                adjusted_price,
                outsideRth=True,
                tif='DAY'
            )
            trade = ib.placeOrder(contract, main_order)
            await ibkr_client.get_account_balance()
            logging.info(
                f"Order placed: SELL {current_position} {symbol} at ${adjusted_price:.2f} "
                f"(outside RTH enabled, TIF=DAY)"
            )
            # Log the trade with extra info indicating the source.
            log_trade(symbol, "SELL", current_position, adjusted_price, trade.order.orderId, extra_info="OrderManagement.place_order SELL")
            # Monitor the order.
            symbol_monitor = SymbolMonitor(
                symbol=symbol,
                total_quantity=current_position,
                entry_price=adjusted_price,
                tick_size=tick_size,
                starting_ema_values=starting_ema_values,
                ibkr_client=ibkr_client
            )
            asyncio.create_task(symbol_monitor.monitor_order(trade))
            # On successful sell, reset internal tracker.
            open_positions[symbol] = 0
            return {"status": "Order placed successfully", "order_id": trade.order.orderId}

        elif action.upper() == "BUY":
            # Check if a position is already held.
            if open_positions.get(symbol, 0) > 0:
                logging.info(f"Buy order for {symbol} skipped – position already held (quantity: {open_positions[symbol]}).")
                return {"status": f"Position in {symbol} already open."}

            if ibkr_client.available_equity is None:
                logging.error("Available equity not fetched yet.")
                return {"error": "Available equity not available"}

            if price is None:
                contract = Stock(symbol, "SMART", "USD")
                ticker = ib.reqMktData(contract, '', False, False)
                total_wait = 0.0
                while ticker.last is None or (isinstance(ticker.last, float) and math.isnan(ticker.last)):
                    await asyncio.sleep(0.1)
                    total_wait += 0.1
                    if total_wait > 5:
                        logging.error(f"Fetched latest price for {symbol} is still NaN after waiting.")
                        return {"error": "Fetched latest price is NaN"}
                price = ticker.last
                logging.info(f"Fetched latest price for {symbol}: {price}")

            if math.isnan(price):
                logging.error(f"Latest price for {symbol} is NaN. Aborting order placement.")
                return {"error": "Fetched latest price is NaN"}

            adjusted_price = round_to_tick(price * (1 + buffer), tick_size)
            if math.isnan(adjusted_price):
                logging.error(f"Computed adjusted price for {symbol} is NaN. Aborting order placement.")
                return {"error": "Adjusted price is NaN"}

            # Determine quantity to buy using a strict limit of 10% of available equity per symbol.
            async with ibkr_client.state_lock:
                quantity = int((ibkr_client.available_equity * 0.10) // adjusted_price)
            
            if quantity <= 0:
                logging.error(f"Calculated quantity for {symbol} is 0. Insufficient equity to place a BUY order.")
                return {"error": "Insufficient equity to place order"}

            contract = Stock(symbol, "SMART", "USD")
            main_order = LimitOrder(
                "BUY",
                quantity,
                adjusted_price,
                outsideRth=True,
                tif='DAY'
            )
            trade = ib.placeOrder(contract, main_order)
            await ibkr_client.get_account_balance()
            logging.info(
                f"Order placed: BUY {quantity} {symbol} at ${adjusted_price:.2f} "
                f"(outside RTH enabled, TIF=DAY)"
            )
            # Log the trade with source information.
            log_trade(symbol, "BUY", quantity, adjusted_price, trade.order.orderId, extra_info="OrderManagement.place_order BUY")
            symbol_monitor = SymbolMonitor(
                symbol=symbol,
                total_quantity=quantity,
                entry_price=adjusted_price,
                tick_size=tick_size,
                starting_ema_values=starting_ema_values,
                ibkr_client=ibkr_client
            )
            asyncio.create_task(symbol_monitor.monitor_order(trade))
            # On successful buy, update the internal tracker.
            open_positions[symbol] = quantity
            return {"status": "Order placed successfully", "order_id": trade.order.orderId}

        else:
            logging.error(f"Invalid action: {action}")
            return {"error": "Invalid action specified"}

    except Exception as e:
        logging.error(f"Error placing order for {symbol}: {e}", exc_info=True)
        return {"error": "Failed to place order", "details": str(e)}


async def close_all_positions(symbol, tick_size, price):
    """
    Close all open positions for a given symbol.
    This function cancels active orders and places SELL orders for all open shares.
    """
    ib = ibkr_client.ib
    symbol = symbol.upper()
    try:
        await ibkr_client.ensure_connection()
        buffer = PRE_MARKET_BUFFER
        await ibkr_client.cancel_active_orders(symbol)
        if open_positions.get(symbol, 0) <= 0:
            logging.info(f"No open position in {symbol} to close.")
            return {"status": f"No position in {symbol} to close."}
        quantity = open_positions[symbol]
        limit_price = round_to_tick(price * (1 - buffer), tick_size)
        contract = Stock(symbol, "SMART", "USD")
        sell_order = LimitOrder("SELL", quantity, limit_price, outsideRth=True, tif='DAY')
        trade = ib.placeOrder(contract, sell_order)
        logging.info(f"Close order placed for {symbol}: OrderId {sell_order.orderId}")
        # Log the trade with extra info indicating the source.
        from trade_logger import log_trade
        log_trade(symbol, "SELL", quantity, limit_price, sell_order.orderId, extra_info="OrderManagement.close_all_positions")
        asyncio.create_task(monitor_simple_sell_order(trade, symbol))
        open_positions[symbol] = 0
        return {"status": "Success", "message": f"Positions for {symbol} closed."}
    except Exception as e:
        logging.error(f"Error closing positions for {symbol}: {e}", exc_info=True)
        return {"status": "Error", "message": str(e)}

async def monitor_simple_sell_order(trade, symbol):
    """
    Monitor a sell order until it is filled or cancelled.
    """
    try:
        while not trade.isDone():
            await asyncio.sleep(1)
        logging.info(f"[{symbol}] Sell order {trade.order.orderId} completed with status {trade.orderStatus.status}")
    except Exception as e:
        logging.error(f"[{symbol}] Error monitoring sell order {trade.order.orderId}: {e}", exc_info=True)
