# clean_price_data.py

import pandas as pd
from sqlalchemy import create_engine, text

# Connect to the SQLite database
engine = create_engine('sqlite:///price_data.db')

# Load all data from the price_data table
df = pd.read_sql("SELECT * FROM price_data", engine)

# --- Cleaning Steps ---

# 1. Convert DateTime column to datetime objects and sort by DateTime
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values(by='DateTime')

# 2. Remove duplicate records (if any) based on DateTime and Symbol
df.drop_duplicates(subset=['DateTime', 'Symbol'], inplace=True)

# 3. Drop rows that might still contain missing values (optional)
df.dropna(inplace=True)

# 4. Filter data to only include rows where the time is between 1:00 AM and 4:00 PM.
# (Assuming we want to keep timestamps with hours >= 1 and < 16)
df = df[(df['DateTime'].dt.hour >= 1) & (df['DateTime'].dt.hour < 16)]

# 5. (Optional) Filter or adjust columns that are relevant for your ML model.
selected_columns = [
    'DateTime', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 
    'SMA', 'RSI', 'RollingMean', 'UpperBand', 'LowerBand',
    'MACD', 'MACD_Signal', 'MACD_Histogram', 'ATR', 'Returns', 
    'Volatility', 'AvgVolume', 'RelativeVolume', 'NextClose', 'PriceMovement'
]
df_ml = df[selected_columns].copy()

# 6. (Optional) Reset the index
df_ml.reset_index(drop=True, inplace=True)

# --- Exporting the Cleaned Data ---

# Save the cleaned data to a new CSV file for ML use
df_ml.to_csv("clean_price_data.csv", index=False)
print("Cleaned data has been exported to clean_price_data.csv")

# Alternatively, if you prefer to store the cleaned data in a new table within the same database:
df_ml.to_sql("clean_price_data", engine, if_exists="replace", index=False)
print("Cleaned data has been stored in the 'clean_price_data' table within price_data.db")
