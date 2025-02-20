
# trade_logger.py
# ---------------
# This module is responsible for logging all executed trades into a CSV file. It records details such as:
#   - Timestamp of the trade
#   - Symbol, action (BUY/SELL), quantity, and price
#   - Order ID and any extra information related to the trade
# Trades are appended to a shared CSV log file, facilitating audit trails and post-trade analysis.

import csv
import os
from datetime import datetime

# Define the shared directory path (adjust if needed)
SHARED_DIR = "/app/shared"
LOG_FILE = os.path.join(SHARED_DIR, "trade_log.csv")

def log_trade(symbol, action, quantity, price, order_id, extra_info=""):
    """
    Append a new row to the trade log CSV.
    Columns: Timestamp, Symbol, Action, Quantity, Price, OrderID, ExtraInfo
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Symbol", "Action", "Quantity", "Price", "OrderID", "ExtraInfo"])
        writer.writerow([timestamp, symbol, action, quantity, price, order_id, extra_info])
