import requests

url = 'https://hound-finer-mite.ngrok-free.app/webhook'
data = {
    'symbol': 'ocea',
    'action': 'BUY',
    'price': 0.2090,
    'ema3': 0.1955,
    'ema9': 0.1935,
    'ema12': 0.1925
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())