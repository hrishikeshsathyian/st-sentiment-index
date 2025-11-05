import yfinance as yf
import pandas as pd

# STI ticker on Yahoo Finance is ^STI
sti = yf.Ticker("^STI")

# Fetch historical data (daily prices)
# You can adjust start and end dates
data = sti.history(start="2020-01-01", end="2025-10-22")

# Reset index to have Date as a column
data = data.reset_index()

# Save to CSV
data.to_csv("../dataset/sti_historical_data.csv", index=False)
