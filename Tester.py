import pandas as pd

df = pd.read_csv("clean_price_data.csv")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df = df.sort_values('DateTime').reset_index(drop=True)
df['Close_diff'] = df['Close'].diff().abs()
df['Close_pct_diff'] = df['Close'].pct_change().abs() * 100  # in %
outliers = df.sort_values('Close_diff', ascending=False).head(10)
print(outliers[['DateTime', 'Close', 'Close_diff', 'Close_pct_diff']])


# Basic info
print(df.info())           # Check columns, types, non-null counts
print(df.describe())       # Check min, max, mean, etc.
