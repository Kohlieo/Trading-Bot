import asyncio
import logging
import os
import signal
import sys
import threading
from quart import Quart, request, jsonify
import ib_insync
from ib_insync import IB, Stock, LimitOrder

# Initialize logging with exception tracebacks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Patch the asyncio event loop
ib_insync.util.patchAsyncio()

# Quart setup
app = Quart(__name__)

# IBKR connection instance
ib = IB()

# Variables to store shared state
available_equity = None
state_lock = threading.Lock()

# Tick size cache file path for persistence
TICK_SIZE_FILE = 'tick_sizes.txt'

# Trailing stop parameters
take_profit_percentage = 0.05  # 5% gain (if needed in future)
initial_stop_loss_percentage = 0.02  # 2% initial stop loss
trailing_stop_loss_percentage = 0.03  # 3% trailing stop
stop_loss_fixed_trigger = 0.02  # 1% above entry price where stop loss becomes fixed
pre_market_buffer = 0.005  # 0.5% buffer for pre-market volatility

def load_tick_sizes():
    """
    Load tick sizes from the persistent text file.
    """
    tick_sizes = {}
    if os.path.exists(TICK_SIZE_FILE):
        with open(TICK_SIZE_FILE, 'r') as f:
            for line in f:
                symbol, tick_size = line.strip().split(',')
                tick_sizes[symbol] = float(tick_size)
    return tick_sizes

def save_tick_sizes(tick_sizes):
    """
    Save tick sizes to the persistent text file.
    """
    with open(TICK_SIZE_FILE, 'w') as f:
        for symbol, tick_size in tick_sizes.items():
            f.write(f"{symbol},{tick_size}\n")

# Load initial tick sizes into the cache
tick_size_cache = load_tick_sizes()

def round_to_tick(value, tick_size):
    """
    Round the given value to the nearest tick size.

    Args:
        value (float): The value to round.
        tick_size (float): The tick size to round to.

    Returns:
        float: The value rounded to the nearest tick size.
    """
    return round(value / tick_size) * tick_size

async def connect_ibkr():
    """
    Async function to connect to IBKR.
    """
    try:
        if not ib.isConnected():
            logging.info("Attempting to connect to IBKR...")
            host = "host.docker.internal"  # Adjust as needed
            port = 7496  # Use port 7496 for paper trading
            client_id = 123
            logging.info(f"Connecting to {host}:{port} with client ID {client_id}")
            await ib.connectAsync(host, port, clientId=client_id, timeout=15)
            logging.info("Successfully connected to IBKR.")
    except Exception as e:
        logging.error(f"Failed to connect to IBKR at {host}:{port}: {e}", exc_info=True)
        # Implement retry logic or exit gracefully as needed

async def ensure_connection():
    """
    Ensure that the connection to IBKR is active.
    """
    if not ib.isConnected():
        await connect_ibkr()

async def get_tick_size(symbol):
    """
    Fetch the tick size for the given symbol from IBKR or cache.

    Args:
        symbol (str): The ticker symbol.

    Returns:
        float: The tick size for the symbol.
    """
    if symbol in tick_size_cache:
        return tick_size_cache[symbol]
    try:
        await ensure_connection()
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMarketDataType(1)  # 1 - Live, includes pre-market data

        # Request tick size and other market data
        ticker = ib.reqMktData(contract, '', False, False)

        # Wait for the data to arrive
        while not ticker.priceIncrement:
            await asyncio.sleep(0.1)

        tick_size = ticker.priceIncrement
        logging.info(f"Fetched tick size for {symbol}: {tick_size}")

        # Cache the tick size and save to file
        tick_size_cache[symbol] = tick_size
        save_tick_sizes(tick_size_cache)

        return tick_size
    except Exception as e:
        logging.error(f"Error fetching tick size for {symbol}: {e}", exc_info=True)
        return 0.01  # Default to a common tick size if unable to fetch

async def cancel_active_orders(symbol):
    """
    Cancel active orders for the given symbol.

    Args:
        symbol (str): The ticker symbol.
    """
    try:
        await ensure_connection()
        # Fetch all open orders
        for trade in ib.trades():
            order = trade.order
            if trade.contract.symbol == symbol and not trade.isDone():
                ib.cancelOrder(order)
                # Wait for order cancellation
                while order.orderStatus.status not in ('Cancelled', 'Filled'):
                    await asyncio.sleep(0.1)
                logging.info(f"Cancelled order {order.orderId} for {symbol}.")
    except Exception as e:
        logging.error(f"Error cancelling orders for {symbol}: {e}", exc_info=True)

async def place_order(symbol: str, action: str, price: float = None) -> dict:
    """
    Place an order via IBKR, ensuring proper order handling.
    This version has been modified so that the actual execution price is used for trailing stop logic.
    """
    try:
        await ensure_connection()

        # Fetch the tick size
        tick_size = await get_tick_size(symbol)
        logging.info(f"Tick size for {symbol}: {tick_size}")

        # Set market data type to receive live data suitable for pre-market
        ib.reqMarketDataType(1)

        # Adjust the price for pre-market conditions
        buffer = pre_market_buffer  # 0.5% buffer for pre-market volatility

        if action.upper() == "SELL":
            await cancel_active_orders(symbol)
            result = await close_all_positions(symbol, tick_size, price)
            await get_account_balance()
            logging.info(f"All positions for {symbol} have been closed.")
            return {"status": f"All positions for {symbol} closed successfully.", "details": result}

        if action.upper() == "BUY":
            if available_equity is None:
                logging.error("Available equity not fetched yet.")
                return {"error": "Available equity not available"}

            if price is None:
                contract = Stock(symbol, "SMART", "USD")
                ticker = ib.reqMktData(contract, '', False, False)
                while not ticker.last:
                    await asyncio.sleep(0.1)
                price = ticker.last
                logging.info(f"Fetched latest price for {symbol}: {price}")

            # Calculate an adjusted limit price for order placement only
            adjusted_price = round_to_tick(price * (1 + buffer), tick_size)
            with state_lock:
                quantity = int((available_equity * 0.65) // adjusted_price)
            contract = Stock(symbol, "SMART", "USD")
            main_order = LimitOrder("BUY", quantity, adjusted_price, outsideRth=True, tif='DAY')
            trade = ib.placeOrder(contract, main_order)

            # Update available equity after placing BUY order
            await get_account_balance()

            # Instead of passing the adjusted_price, we now let monitor_order extract the actual fill price.
            asyncio.create_task(monitor_order(trade, symbol, quantity, tick_size))
            logging.info(f"Order placed: BUY {quantity} {symbol} at ${adjusted_price:.2f} (outside RTH enabled, TIF=DAY)")
            return {"status": "Order placed successfully", "order_id": trade.order.orderId}

        logging.error(f"Invalid action: {action}")
        return {"error": "Invalid action specified"}

    except Exception as e:
        logging.error(f"Error placing order: {e}", exc_info=True)
        return {"error": "Failed to place order", "details": str(e)}

async def close_all_positions(symbol, tick_size, price):
    """
    Close all open positions for the given symbol.

    Args:
        symbol (str): The ticker symbol.
        tick_size (float): The tick size for the symbol.
        price (float): The current price of the symbol.

    Returns:
        dict: Result of the operation.
    """
    try:
        await ensure_connection()
        # Define buffer for pre-market conditions
        buffer = pre_market_buffer  # 0.5% buffer for pre-market volatility

        # Cancel any existing orders before placing the new sell order
        await cancel_active_orders(symbol)

        positions = ib.positions()
        for position in positions:
            if position.contract.symbol == symbol and position.position > 0:
                # Adjust price to ensure the order goes through
                limit_price = round_to_tick(price * (1 - buffer), tick_size)  # Slightly below market price

                contract = Stock(symbol, "SMART", "USD")
                sell_order = LimitOrder("SELL", position.position, limit_price, outsideRth=True, tif='DAY')
                trade = ib.placeOrder(contract, sell_order)
                logging.info(f"Close order placed for {symbol}: {trade.order.orderId}.")

                # Monitor and manage the order actively
                asyncio.create_task(monitor_order(trade, symbol, 0, 0, tick_size))

        logging.info(f"All positions for {symbol} have been closed.")
        return {"status": "Success", "message": f"Positions for {symbol} closed."}
    except Exception as e:
        logging.error(f"Error closing positions for {symbol}: {e}", exc_info=True)
        return {"status": "Error", "message": str(e)}

async def monitor_order(trade, symbol, quantity, tick_size):
    """
    Monitor the given order and log status updates.
    For BUY orders, extract the actual fill price from trade.fills and start the trailing stop monitoring
    using that execution price.
    """
    try:
        # Wait until the order is complete
        while not trade.isDone():
            await asyncio.sleep(1)
        logging.info(f"Order {trade.order.orderId} completed with status {trade.orderStatus.status}")

        if trade.orderStatus.status == 'Filled' and trade.order.action.upper() == 'BUY':
            # Small delay to ensure fill details are available
            await asyncio.sleep(0.1)
            if trade.fills:
                actual_fill_price = trade.fills[-1].execution.price
                logging.info(f"Actual fill price for {symbol}: {actual_fill_price}")
            else:
                # Fallback: use the order's limit price if no fills are recorded
                actual_fill_price = trade.order.lmtPrice
                logging.warning(f"No fills recorded for trade {trade.order.orderId}, using limit price {actual_fill_price}")

            # Start monitoring the price using the actual fill price as entry
            asyncio.create_task(monitor_price(symbol, quantity, actual_fill_price, tick_size))
    except Exception as e:
        logging.error(f"Error monitoring order {trade.order.orderId}: {e}", exc_info=True)

async def monitor_price(symbol, quantity, entry_price, tick_size):
    """
    Monitor the price for trailing stop-loss implementation.

    Args:
        symbol (str): The ticker symbol.
        quantity (int): Quantity of the position.
        entry_price (float): Entry price at which the position was opened.
        tick_size (float): The tick size for the symbol.
    """
    try:
        contract = Stock(symbol, "SMART", "USD")
        ib.reqMarketDataType(1)  # Ensure live data
        ticker = ib.reqMktData(contract, '', False, False)

        highest_price = entry_price
        stop_loss_price = round_to_tick(entry_price * (1 - initial_stop_loss_percentage), tick_size)
        trailing = True  # Flag to indicate if trailing is active

        logging.info(f"Monitoring {symbol} - Entry Price: ${entry_price:.2f}, Initial Stop-Loss: ${stop_loss_price:.2f}")
        print(f"Initial Stop-Loss for {symbol}: ${stop_loss_price:.2f}")

        while True:
            await asyncio.sleep(1)
            current_price = ticker.last if ticker.last else ticker.close
            if current_price is None:
                continue  # Wait for price data

            if trailing:
                if current_price > highest_price:
                    highest_price = current_price
                    # Update stop-loss to trail 2% below the highest price
                    stop_loss_price = round_to_tick(highest_price * (1 - trailing_stop_loss_percentage), tick_size)
                    logging.info(f"New high for {symbol}: ${highest_price:.2f}, Trailing Stop-Loss updated to ${stop_loss_price:.2f}")
                    print(f"Trailing Stop-Loss for {symbol} updated to: ${stop_loss_price:.2f}")

                # Check if price has reached 1% above entry price
                if current_price >= entry_price * (1 + stop_loss_fixed_trigger):
                    trailing = False  # Stop trailing
                    # Fix the stop-loss price at current level
                    stop_loss_price = round_to_tick(current_price * (1 - trailing_stop_loss_percentage), tick_size)
                    logging.info(f"{symbol} has reached 1% above entry price. Stop-loss fixed at ${stop_loss_price:.2f}")
                    print(f"Stop-Loss for {symbol} fixed at: ${stop_loss_price:.2f}")

            if current_price <= stop_loss_price:
                logging.info(f"{symbol} has reached the stop-loss price of ${stop_loss_price:.2f}. Initiating sell order.")
                await place_sell_order(symbol, quantity, current_price, tick_size)
                break

    except Exception as e:
        logging.error(f"Error monitoring price for {symbol}: {e}", exc_info=True)


async def place_sell_order(symbol, quantity, current_price, tick_size):
    """
    Place a sell order for the given symbol at the current price.

    Args:
        symbol (str): The ticker symbol.
        quantity (int): Quantity to sell.
        current_price (float): Current market price.
        tick_size (float): Tick size for the symbol.
    """
    try:
        await ensure_connection()

        contract = Stock(symbol, "SMART", "USD")
        ib.reqMarketDataType(1)  # Ensure live data

        # Adjust price slightly with buffer for better execution
        buffer = pre_market_buffer
        limit_price = round_to_tick(current_price * (1 - buffer), tick_size)
        sell_order = LimitOrder("SELL", quantity, limit_price, outsideRth=True, tif='DAY')
        sell_trade = ib.placeOrder(contract, sell_order)

        # Update available equity after placing SELL order
        await get_account_balance()

        logging.info(f"Placed SELL order for {quantity} {symbol} at ${limit_price:.2f}")

        # Monitor the sell order
        asyncio.create_task(monitor_order(sell_trade, symbol, 0, 0, tick_size))

    except Exception as e:
        logging.error(f"Error placing sell order for {symbol}: {e}", exc_info=True)

@app.route('/webhook', methods=['POST'])
async def webhook():
    """
    Handle incoming webhook requests to place orders.
    """
    data = await request.get_json()
    logging.debug(f"Received data: {data}")

    symbol = data.get('symbol')
    action = data.get('action')
    price = data.get('price')

    if not symbol or not action or price is None:
        return jsonify({"error": "Invalid input"}), 400

    valid_actions = {'BUY', 'SELL'}
    if action.upper() not in valid_actions:
        return jsonify({"error": "Invalid action"}), 400

    try:
        price = float(price)
    except ValueError:
        return jsonify({"error": "Invalid price value"}), 400

    # Start the order placement task
    asyncio.create_task(place_order(symbol, action, price))

    return jsonify({"message": "Order request is being processed."}), 202

async def get_account_balance():
    """
    Fetches and sets the available equity.
    """
    global available_equity
    try:
        await ensure_connection()
        if not ib.isConnected():
            logging.error("Cannot get account balance because IBKR is not connected.")
            return
        account_summary = await ib.accountSummaryAsync()

        for account_value in account_summary:
            if account_value.tag == 'AvailableFunds':
                with state_lock:
                    available_equity = float(account_value.value)
                logging.info(f"Available Equity: {available_equity}")
                break
    except Exception as e:
        logging.error(f"Error getting account balance: {e}", exc_info=True)

async def sync_positions_with_ibkr():
    """
    Log current IBKR positions at startup.
    """
    try:
        await ensure_connection()
        positions = ib.positions()
        for position in positions:
            symbol = position.contract.symbol
            logging.info(f"Position: {symbol}, Quantity: {position.position}, Avg Cost: {position.avgCost}")
    except Exception as e:
        logging.error(f"Error syncing positions with IBKR: {e}", exc_info=True)

def signal_handler(sig, frame):
    """
    Handle shutdown signals to gracefully disconnect from IBKR.
    """
    logging.info("Shutting down...")
    if ib.isConnected():
        ib.disconnect()
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """
    Main function to start the application.
    """
    await connect_ibkr()
    await get_account_balance()
    await sync_positions_with_ibkr()

    # Removed the periodic update task

    # Run the Quart app
    await app.run_task(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    asyncio.run(main())
