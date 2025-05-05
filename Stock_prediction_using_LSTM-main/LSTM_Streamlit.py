# Import Libraries
import yfinance as yf
import ta
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Load the Stock Data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Adj Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'].squeeze()).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data['Adj Close'].squeeze()).ema_indicator()
    data.dropna(inplace=True)
    return data

# User Inputs
stock_symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2023-12-31"

data = load_stock_data(stock_symbol, start_date, end_date)
print(data.head())

# Step 2: Plot the stock price, RSI, and EMA
plt.figure(figsize=(14, 12))

# Plot Adjusted Close Price
plt.subplot(3, 1, 1)
plt.plot(data['Adj Close'], label=f"{stock_symbol} Adjusted Close Price")
plt.title(f"{stock_symbol} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

# Plot RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(y=70, color='r', linestyle='--', label="Overbought (70)")
plt.axhline(y=30, color='g', linestyle='--', label="Oversold (30)")
plt.title("Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()

# Plot EMA
plt.subplot(3, 1, 3)
plt.plot(data['EMA'], label='EMA', color='purple')
plt.title("Exponential Moving Average (EMA)")
plt.xlabel("Date")
plt.ylabel("EMA Value")
plt.legend()

# Show plots
plt.tight_layout()
plt.show()
