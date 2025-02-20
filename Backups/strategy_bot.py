import asyncio
import logging
import pandas as pd
import pandas_ta as ta
import time
from ib_insync import IB, Stock, util, LimitOrder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize IBKR connection
ib = IB()

# Symbol to monitor
symbol = 'AMOD'  # Change this to the symbol you wish to monitor
contract = Stock(symbol, 'SMART', 'USD')

# Global variables
symbol_data = pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
position = None  # Tracks the open position
available_equity = None  # To be fetched from account summary

# Strategy parameters (from your TradingView script)
base_price_jump = 0.52 / 100      # Base Price Jump (%)
volume_threshold = 10000          # Volume Threshold
ema_length = 16                   # EMA Length for exit
stop_loss_cents = 4.0 / 100       # Stop Loss in dollars

async def connect_ibkr():
    """Connect to the Interactive Brokers API."""
    try:
        if not ib.isConnected():
            await ib.connectAsync('127.0.0.1', 7497, clientId=123)
            logging.info("Connected to IBKR.")
    except Exception as e:
        logging.error(f"Connection error: {e}")
        raise

async def get_account_balance():
    """Fetch the available equity for trading."""
    global available_equity
    try:
        account_summary = await ib.accountSummaryAsync()
        for item in account_summary:
            if item.tag == 'AvailableFunds' and item.currency == 'USD':
                available_equity = float(item.value)
                logging.info(f"Available equity: ${available_equity:.2f}")
                break
    except Exception as e:
        logging.error(f"Error fetching account balance: {e}")

def has_open_position():
    """Check if there's an open position."""
    return position is not None and position.position != 0

def get_position_avg_price():
    """Get the average price of the open position."""
    if position:
        return position.avgCost
    return 0.0

async def update_position():
    """Update the position variable with the current position."""
    global position
    try:
        positions = ib.positions()
        position = None
        for pos in positions:
            if pos.contract.symbol == symbol:
                position = pos
                break
    except Exception as e:
        logging.error(f"Error updating position: {e}")

def round_to_tick(value, tick_size):
    """Round a price to the nearest tick size."""
    return round(value / tick_size) * tick_size

async def get_tick_size():
    """Fetch the minimum price increment (tick size) for the symbol."""
    try:
        details = await ib.reqContractDetailsAsync(contract)
        if details and details[0].minTick:
            return details[0].minTick
    except Exception as e:
        logging.error(f"Error fetching tick size: {e}")
    return 0.01  # Default tick size

async def process_market_data():
    """Process market data updates for the contract."""
    logging.info(f"Started monitoring {symbol}")
    tick_size = await get_tick_size()
    
    # Since you have live data, no need to set market data type
    # ib.reqMarketDataType(1)  # Live data is default or can be set explicitly if needed
    
    subscriber = ib.reqMktData(contract, '', False, False)

    while True:
        await asyncio.sleep(1)  # Adjust as needed
        try:
            last_price = subscriber.last
            last_volume = subscriber.volume
            if last_price is None or last_volume is None:
                continue  # Wait for valid data

            data = {
                'datetime': pd.to_datetime('now'),
                'open': subscriber.open,
                'high': subscriber.high,
                'low': subscriber.low,
                'close': last_price,
                'volume': last_volume
            }
            global symbol_data
            symbol_data = symbol_data.append(data, ignore_index=True)
            symbol_data = symbol_data.tail(100)  # Keep last 100 entries

            # Calculate indicators and evaluate strategy
            data_point = calculate_indicators()
            if data_point is not None:
                await evaluate_strategy(data_point, tick_size)
        except Exception as e:
            logging.error(f"Error processing market data: {e}")

def calculate_indicators():
    """Calculate indicators based on the strategy."""
    df = symbol_data.copy()
    if len(df) < max(20, ema_length):
        return None  # Not enough data

    # Calculate indicators
    df['emaExit'] = ta.ema(df['close'], length=ema_length)
    df['emaShort'] = ta.ema(df['close'], length=6)
    df['emaLong'] = ta.ema(df['close'], length=20)
    df['priceChange'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['isBullishTrend'] = df['emaShort'] > df['emaLong']

    return df.iloc[-1]

async def evaluate_strategy(data_point, tick_size):
    """Evaluate the strategy and place orders if conditions are met."""
    if not has_open_position():
        # Entry conditions
        if len(symbol_data) < 2:
            return  # Not enough data to calculate previous close

        previous_close = symbol_data['close'].iloc[-2]
        scaled_price_jump = base_price_jump * previous_close
        price_change = data_point['priceChange']
        volume = data_point['volume']
        is_bullish_trend = data_point['isBullishTrend']

        entry_condition = (
            price_change >= scaled_price_jump and
            volume >= volume_threshold and
            is_bullish_trend
        )

        if entry_condition:
            price = data_point['close']
            await place_order('BUY', price, tick_size)
    else:
        # Exit conditions
        position_avg_price = get_position_avg_price()
        stop_loss_price = position_avg_price - stop_loss_cents
        is_prev_four_bars_red = check_prev_four_bars_red()
        close_price = data_point['close']
        ema_exit = data_point['emaExit']

        exit_condition = (
            close_price < ema_exit or
            close_price < stop_loss_price or
            is_prev_four_bars_red
        )

        if exit_condition:
            await place_order('SELL', close_price, tick_size)

def check_prev_four_bars_red():
    """Check if the previous four bars are red (close < open)."""
    df = symbol_data
    if len(df) < 5:
        return False  # Not enough data
    for i in range(1, 5):
        if df['close'].iloc[-i] >= df['open'].iloc[-i]:
            return False
    return True

async def place_order(action, price, tick_size):
    """Place an order for the symbol."""
    try:
        if action.upper() == 'BUY':
            quantity = calculate_quantity(price)
            if quantity <= 0:
                logging.error(f"Calculated quantity is zero or negative.")
                return
            adjusted_price = round_to_tick(price + 0.03, tick_size)
            parent_order = LimitOrder('BUY', quantity, adjusted_price, outsideRth=True)
            trade = ib.placeOrder(contract, parent_order)
            logging.info(f"Placed BUY order at ${adjusted_price:.2f} for {quantity} shares.")

            # Wait for order fill using callback
            trade.filledEvent += lambda trade, contract=contract, quantity=quantity, tick_size=tick_size: \
                asyncio.create_task(on_order_filled(trade, contract, quantity, tick_size))
        elif action.upper() == 'SELL':
            if position and position.position > 0:
                quantity = position.position
                adjusted_price = round_to_tick(price - 0.03, tick_size)
                order = LimitOrder('SELL', quantity, adjusted_price, outsideRth=True)
                trade = ib.placeOrder(contract, order)
                logging.info(f"Placed SELL order at ${adjusted_price:.2f} for {quantity} shares.")
            else:
                logging.info("No open position to sell.")
        else:
            logging.error(f"Invalid action: {action}")
    except Exception as e:
        logging.error(f"Error placing order: {e}")

async def on_order_filled(trade, contract, quantity, tick_size):
    """Callback function when an order is filled."""
    if trade.orderStatus.status == 'Filled':
        avg_fill_price = trade.orderStatus.avgFillPrice
        logging.info(f"Order filled at ${avg_fill_price:.2f}, placing OCO orders.")
        await place_oco_orders(contract, quantity, avg_fill_price, tick_size)
        await update_position()

async def place_oco_orders(contract, quantity, avg_fill_price, tick_size):
    """Place OCO orders for profit-taking and stop-loss."""
    oca_group = f"OCA_{symbol}_{int(time.time())}"

    # Calculate prices
    stop_loss_price = avg_fill_price - stop_loss_cents
    take_profit_price = avg_fill_price * 1.08
    stop_loss_price = round_to_tick(stop_loss_price, tick_size)
    take_profit_price = round_to_tick(take_profit_price, tick_size)

    # Create OCO orders
    stop_loss_order = LimitOrder(
        'SELL',
        quantity,
        stop_loss_price,
        outsideRth=True,
        ocaGroup=oca_group,
        ocaType=1
    )

    take_profit_order = LimitOrder(
        'SELL',
        quantity,
        take_profit_price,
        outsideRth=True,
        ocaGroup=oca_group,
        ocaType=1
    )

    ib.placeOrder(contract, stop_loss_order)
    ib.placeOrder(contract, take_profit_order)
    logging.info(f"Placed OCO orders: Stop Loss at ${stop_loss_price:.2f}, Take Profit at ${take_profit_price:.2f}")

def calculate_quantity(price):
    """Calculate the order quantity based on available equity."""
    if available_equity is None:
        logging.error("Available equity not set.")
        return 0
    allocation = available_equity * 0.35  # 35% of available equity
    quantity = int(allocation / price)
    return quantity

async def main():
    """Main function to run the bot."""
    await connect_ibkr()
    await get_account_balance()
    await update_position()
    ib.qualifyContracts(contract)

    # Start market data processing
    await process_market_data()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    finally:
        if ib.isConnected():
            ib.disconnect()
