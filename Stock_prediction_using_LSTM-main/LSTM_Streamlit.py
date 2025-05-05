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
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error(f"No data found for ticker: {ticker}. Please check the symbol and date range.")
        st.stop()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if 'Adj Close' not in data.columns:
        st.error(f"Missing expected column in data: 'Adj Close'. Available columns: {data.columns.tolist()}")
        st.stop()

    data['Return'] = data['Adj Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close']).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data['Adj Close']).ema_indicator()
    data.dropna(inplace=True)
    return data

# Step 2: Prepare the Dataset
def prepare_data(data, lookback=14):
    features = ['Adj Close', 'Volume', 'RSI', 'EMA']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])  # Target is 'Adj Close'
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test, scaler, features

# Step 3: Model Loading and Prediction
def load_and_predict(model_file, X_test, scaler, features, data):
    model = load_model(model_file, custom_objects={'mse': MeanSquaredError()})
    y_pred = model.predict(X_test)

    def rescale(data, predictions):
        dummy_features = np.zeros((len(predictions), len(features) - 1))
        rescaled = scaler.inverse_transform(np.concatenate([predictions, dummy_features], axis=1))
        return rescaled[:, 0]

    y_pred_rescaled = rescale(data[features], y_pred)
    return y_pred_rescaled

# Step 4: Streamlit Web App
st.title("Stock Price Prediction with LSTM")
st.markdown("This app uses an LSTM model to predict stock prices based on historical data.")

stock_symbol = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if stock_symbol:
    data = load_stock_data(stock_symbol, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)
    
    model_path = '/Users/manojkumarbollineni/Desktop/LSTM/lstm_stock_model.h5'
    try:
        y_pred_rescaled = load_and_predict(model_path, X_test, scaler, features, data)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="True Prices", alpha=0.7)
    plt.plot(y_pred_rescaled, label="Predicted Prices", alpha=0.7)
    plt.title(f"True vs Predicted Prices for {stock_symbol}")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

    st.write(f"Predicted Prices for {stock_symbol} (Last 5 predictions):")
    st.write(y_pred_rescaled[-5:])
