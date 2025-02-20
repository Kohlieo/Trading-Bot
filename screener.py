import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import progressbar
import time
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Your custom FinViz screener URL
CUSTOM_URL = ("https://finviz.com/screener.ashx?v=151&f=news_date_sinceyesterday,sh_avgvol_o50,sh_float_u20,sh_price_u20,sh_relvol_o5,ta_change_u10&ft=4&o=-gap&ar=180&c=0,1,25,30,84,36,60,61,64,67,65")

# Update OUTPUT_FILE path to point to the shared volume.
OUTPUT_FILE = "C:/SharedData/rl_trader_tickers.json"

def fetch_finviz_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/90.0.4430.93 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    logging.info("Fetching FinViz screener page...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def parse_all_tables(html):
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    dataframes = []
    for table in tables:
        try:
            # Wrap the table HTML with StringIO to avoid warnings.
            df = pd.read_html(StringIO(str(table)))[0]
            dataframes.append(df)
        except Exception as e:
            continue
    return dataframes

def main():
    html_content = fetch_finviz_data(CUSTOM_URL)
    tables = parse_all_tables(html_content)
    logging.info(f"Found {len(tables)} tables on the page.")
    
    # Use table 7 as in your original script.
    if len(tables) > 8:
        df = tables[8]
        logging.info("Using table 8 as the data table.")
        tickers = df["Ticker"].head(5).tolist() if "Ticker" in df.columns else df.iloc[:,0].head(5).tolist()
        logging.info(f"Extracted tickers: {tickers}")
    else:
        logging.error("Table 8 was not found in the parsed tables.")
        tickers = []

    # Ensure the output directory exists.
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(tickers, f)
        logging.info(f"Updated ticker file {OUTPUT_FILE} with: {tickers}")
    except Exception as e:
        logging.error(f"Error writing to {OUTPUT_FILE}: {e}")

if __name__ == "__main__":
    while True:
        main()
        logging.info("Sleeping for 3.1 minutes...")
        time.sleep(186)  # 3.1 minutes = 186 seconds
