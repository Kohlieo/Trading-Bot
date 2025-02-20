# webhook_handler.py

from quart import Quart, request, jsonify
import logging
import asyncio
import os
from order_management import place_order

app = Quart(__name__)

# Remove the authentication token since we're eliminating authorization
# WEBHOOK_AUTH_TOKEN = os.getenv('WEBHOOK_AUTH_TOKEN')

@app.route('/webhook', methods=['POST'])
async def webhook():
    data = await request.get_json()
    logging.debug(f"Received data: {data}")

    symbol = data.get('symbol')
    action = data.get('action')
    price = data.get('price')
    starting_ema3 = data.get('ema3')
    starting_ema9 = data.get('ema9')
    starting_ema12 = data.get('ema12')

    if not symbol or not action or price is None:
        logging.error("Invalid input data")
        return jsonify({"error": "Invalid input"}), 400

    valid_actions = {'BUY', 'SELL'}
    if action.upper() not in valid_actions:
        logging.error("Invalid action specified")
        return jsonify({"error": "Invalid action"}), 400

    try:
        price = float(price)
        starting_ema3 = float(starting_ema3) if starting_ema3 else None
        starting_ema9 = float(starting_ema9) if starting_ema9 else None
        starting_ema12 = float(starting_ema12) if starting_ema12 else None
    except ValueError:
        logging.error("Invalid numerical value in input")
        return jsonify({"error": "Invalid numerical value in input"}), 400

    ema_values = (starting_ema3, starting_ema9, starting_ema12)

    # Ensure `place_order` runs without blocking
    asyncio.create_task(
        place_order(symbol, action, price, starting_ema_values=ema_values)
    )

    return jsonify({"message": "Order request is being processed."}), 202
