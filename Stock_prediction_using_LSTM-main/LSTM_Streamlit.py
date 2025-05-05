import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Step 1: Load the Stock Data
def load_stock_data(stock_symbol, start, end):
    data = yf.download(stock_symbol, start=start, end=end)

    if data.empty:
        st.error("No data found. Please check the stock symbol or date range.")
        return None

    # ✅ Use 'Adj Close' if available, else fallback to 'Close'
    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data.rename(columns={price_col: 'Price'}, inplace=True)

    # Add technical indicators
    data['Volume'] = data['Volume'].fillna(method='ffill')
    data['RSI'] = ta.momentum.RSIIndicator(data['Price'], window=14).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data['Price'], window=14).ema_indicator()
    
    data.dropna(inplace=True)
    return data

# Step 2: Prepare the Dataset
def prepare_data(data, lookback=14):
    features = ['Price', 'Volume', 'RSI', 'EMA']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Target is the first column: 'Price'
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    
    # Split data into training and testing
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, features

# Step 3: Model Loading and Prediction
def load_and_predict(model_file, X_test, scaler, features):
    model = load_model(model_file, custom_objects={'mse': MeanSquaredError()})
    y_pred = model.predict(X_test)

    def rescale(predictions):
        dummy = np.zeros((len(predictions), len(features)-1))
        padded = np.concatenate([predictions, dummy], axis=1)
        rescaled = scaler.inverse_transform(padded)
        return rescaled[:, 0]

    y_pred_rescaled = rescale(y_pred)
    return y_pred_rescaled

# Step 4: Streamlit Web App
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("This app uses an LSTM model to predict stock prices based on historical data.")

# User Input
stock_symbol = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

data = load_stock_data(stock_symbol, start_date, end_date)
if data is None:
    st.stop()

X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)

# Load model and predict
model_path = '/Users/manojkumarbollineni/Desktop/LSTM/lstm_stock_model.h5'
try:
    y_pred_rescaled = load_and_predict(model_path, X_test, scaler, features)
except Exception as e:
    st.error(f"Error loading model or predicting: {e}")
    st.stop()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="True Prices", alpha=0.7)
plt.plot(y_pred_rescaled, label="Predicted Prices", alpha=0.7)
plt.title(f"True vs Predicted Prices for {stock_symbol}")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
st.pyplot(plt)

# Show Predictions
st.subheader(f"Predicted Prices for {stock_symbol} (First 5):")
st.write(y_pred_rescaled[:5])
