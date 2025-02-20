import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from polygon import RESTClient
from sklearn.preprocessing import StandardScaler
import pandas_market_calendars as mcal  # Import market calendar

# Configure logging with more informative logs
logging.basicConfig(
    filename='data_farmer.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s'
)

# Load environment variables
load_dotenv()
POLYGON_KEY = os.getenv('POLYGON_KEY')
if not POLYGON_KEY:
    raise Exception("POLYGON_KEY not found in .env file.")

# Initialize Polygon API client
client = RESTClient(POLYGON_KEY)

# Get the NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Read the Excel file and reshape
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the Excel file
excel_file = os.path.join(script_dir, 'top_gainers.xlsx')
# Read the Excel file
top_gainers_df = pd.read_excel(excel_file)

# Ensure 'Date' is in datetime format
top_gainers_df['Date'] = pd.to_datetime(top_gainers_df['Date'])

# Debug: Print columns in the Excel file
print("Columns in the Excel file:")
print(top_gainers_df.columns.tolist())

# Melt the DataFrame to have one symbol per row
# Adjust value_vars to match the actual column names in your Excel file
value_vars = ['symbol 1', 'symbol 2', 'symbol 3', 'symbol 4', 'symbol 5']
top_gainers_df = top_gainers_df.melt(
    id_vars=['Date'],
    value_vars=value_vars,
    var_name='Symbol_Rank',
    value_name='Symbol'
)

# Drop NaN symbols and unnecessary columns
top_gainers_df.dropna(subset=['Symbol'], inplace=True)
top_gainers_df.drop(columns=['Symbol_Rank'], inplace=True)
top_gainers_df.reset_index(drop=True, inplace=True)

# Normalize 'Symbol' strings to uppercase
top_gainers_df['Symbol'] = top_gainers_df['Symbol'].str.upper()

# Debug: Print the DataFrame after melting
print("DataFrame after melting:")
print(top_gainers_df.head())
print(f"Total symbols to process: {len(top_gainers_df)}")

# Create or connect to the database using SQLAlchemy with connection pooling
engine = create_engine('sqlite:///price_data.db', poolclass=QueuePool)

# Create the price_data table if it doesn't exist
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT,
            DateTime TEXT,
            Symbol TEXT COLLATE NOCASE,
            Interval TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            Transactions INTEGER,
            SMA REAL,
            RSI REAL,
            RollingMean REAL,
            UpperBand REAL,
            LowerBand REAL,
            MACD REAL,
            MACD_Signal REAL,
            MACD_Histogram REAL,
            ATR REAL,
            Returns REAL,
            Volatility REAL,
            AvgVolume REAL,
            RelativeVolume REAL,
            NextClose REAL,
            PriceMovement INTEGER,
            UNIQUE (DateTime, Symbol) ON CONFLICT IGNORE
        )
    """))
    print("Checked for 'price_data' table and created it if it did not exist.")

# Ensure indexes are created if they don't exist
with engine.connect() as conn:
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_symbol_date ON price_data (Symbol, Date)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_datetime ON price_data (DateTime)"))
    print("Checked for indexes and created them if they did not exist.")

# Technical indicator functions

def compute_sma(series, window=3):
    """Compute the Simple Moving Average (SMA) for a given series."""
    return series.rolling(window=window).mean()

def compute_rsi(series, window=14):
    """Compute the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(series, window=20, num_std_dev=2):
    """Compute Bollinger Bands for a given series."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std(ddof=0)
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return rolling_mean, upper_band, lower_band

def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """Compute the Moving Average Convergence Divergence (MACD) for a given series."""
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def compute_atr(df, window=14):
    """Compute the Average True Range (ATR) for a given DataFrame."""
    high_low = df['High'] - df['Low']
    high_close_prev = (df['High'] - df['Close'].shift()).abs()
    low_close_prev = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

# Function to check if data already exists for a symbol and date
def data_already_stored(symbol, date):
    """
    Check if data for the given symbol and date already exists in the database.
    """
    query = text("""
        SELECT 1 FROM price_data
        WHERE Symbol = :symbol AND Date = :date
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"symbol": symbol, "date": date.strftime('%Y-%m-%d')}).fetchone()
    return result is not None

# Function to check if a date is a trading day
def is_trading_day(date):
    """
    Check if the given date is a trading day on the NYSE.
    """
    schedule = nyse.valid_days(start_date=date, end_date=date)
    return not schedule.empty

# Function to fetch intraday data with retry logic
def fetch_intraday_data(symbol, date_str, resolution='minute', max_retries=3):
    attempts = 0
    while attempts < max_retries:
        try:
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan=resolution,
                from_=date_str,
                to=date_str,
                limit=50000
            )
            return aggs
        except Exception as e:
            attempts += 1
            logging.error(f"Attempt {attempts}: Error fetching data for {symbol} on {date_str}: {e}")
            time.sleep(5)
    return None

# Function to process, compute indicators, handle data normalization and feature engineering, and store data
def store_intraday_data(symbol, date):
    start_time = time.time()
    date_str = date.strftime('%Y-%m-%d')
    logging.info(f"Fetching data for {symbol} on {date_str}...")
    aggs = fetch_intraday_data(symbol, date_str, resolution='minute')
    if not aggs or len(aggs) == 0:
        logging.warning(f"No data available for {symbol} on {date_str}.")
        print(f"No data available for {symbol} on {date_str}.")
        return
    
    # Convert fetched aggs to DataFrame
    data_list = [agg.__dict__ for agg in aggs]
    if not data_list:
        logging.warning(f"No records returned for {symbol} on {date_str}.")
        print(f"No records returned for {symbol} on {date_str}.")
        return
    
    df = pd.DataFrame(data_list)
    
    # Convert timestamp to datetime
    df['DateTime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Floor the datetime to the nearest second to avoid microsecond differences
    df['DateTime'] = df['DateTime'].dt.floor('S')
    
    # Rename columns for consistency
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'transactions': 'Transactions'
    }, inplace=True)
    
    # Normalize the 'Symbol' to uppercase
    df['Symbol'] = symbol.upper()
    df['Date'] = date.date()
    df['Interval'] = '1min'
    
    # Compute technical indicators
    df['SMA'] = compute_sma(df['Close'], window=3)
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df['RollingMean'], df['UpperBand'], df['LowerBand'] = compute_bollinger_bands(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = compute_macd(df['Close'])
    df['ATR'] = compute_atr(df)
    
    # Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=14).std()
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['RelativeVolume'] = df['Volume'] / df['AvgVolume']
    
    # Label Generation for Machine Learning
    df['NextClose'] = df['Close'].shift(-1)
    df['PriceMovement'] = (df['NextClose'] > df['Close']).astype(int)
    
    # Handle missing values
    df.infer_objects(copy=False)
    df.interpolate(method='linear', inplace=True)
    df = df.bfill()
    
    # Normalize features (if needed)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA', 'RSI', 'RollingMean',
                    'UpperBand', 'LowerBand', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                    'ATR', 'Returns', 'Volatility', 'AvgVolume', 'RelativeVolume']
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Reorder columns for clarity
    df = df[['Date', 'DateTime', 'Symbol', 'Interval', 'Open', 'High', 'Low', 'Close', 'Volume', 'Transactions',
             'SMA', 'RSI', 'RollingMean', 'UpperBand', 'LowerBand', 'MACD', 'MACD_Signal', 'MACD_Histogram',
             'ATR', 'Returns', 'Volatility', 'AvgVolume', 'RelativeVolume', 'NextClose', 'PriceMovement']]
    
    # Store data in the SQLite database
    try:
        df.to_sql('price_data', con=engine, if_exists='append', index=False, method='multi')
        logging.info(f"Data for {symbol} on {date.date()} stored successfully. Inserted {len(df)} new records.")
        print(f"Data for {symbol} on {date.date()} stored successfully. Inserted {len(df)} new records.")
    except Exception as e:
        logging.error(f"Error storing data for {symbol} on {date.date()}: {e}")
        print(f"Error storing data for {symbol} on {date.date()}: {e}")
    
    end_time = time.time()
    logging.info(f"Time taken for {symbol} on {date.date()}: {end_time - start_time} seconds")

# Main loop: Batch process symbols per date
def main():
    print("Processing the following symbols and dates:")
    print(top_gainers_df)
    print(f"Total symbols to process: {len(top_gainers_df)}")
    
    # Group symbols by date
    grouped = top_gainers_df.groupby('Date')['Symbol'].apply(list).reset_index()
    
    for index, row in grouped.iterrows():
        date = row['Date']
        symbols = row['Symbol']
        
        print(f"\nProcessing date: {date.date()} with symbols: {symbols}")
        
        # List of dates: event date and previous day
        date_list = [date - pd.Timedelta(days=1), date]
          
        for single_date in date_list:
            # Check if single_date is a trading day
            if not is_trading_day(single_date):
                print(f"{single_date.date()} is not a trading day. Skipping.")
                logging.info(f"{single_date.date()} is not a trading day. No data fetched.")
                continue
            
            print(f"\nProcessing date: {single_date.date()}")
            symbols_to_process = []
            
            # Check which symbols need to be processed
            for symbol in symbols:
                if data_already_stored(symbol, single_date):
                    print(f"Data for {symbol} on {single_date.date()} already exists. Skipping.")
                else:
                    symbols_to_process.append(symbol)
            
            # Process each symbol
            for symbol in symbols_to_process:
                print(f"Fetching data for {symbol} on {single_date.date()}...")
                store_intraday_data(symbol, single_date)
                # Respect API rate limits after each symbol
                time.sleep(12)

    # Optionally, test querying the stored data:
    for symbol in top_gainers_df['Symbol'].unique():
        df_query = query_price_data(symbol, "2025-01-01", "2025-12-31")
        print(f"Queried data for {symbol}: {len(df_query)} records")

# Helper function to query data from the database (for ML use later)
def query_price_data(symbol, start_date, end_date):
    query = text("""
        SELECT * FROM price_data
        WHERE Symbol = :symbol AND Date BETWEEN :start_date AND :end_date
        ORDER BY DateTime ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"symbol": symbol.upper(), "start_date": start_date, "end_date": end_date})
    return df

if __name__ == "__main__":
    main()
