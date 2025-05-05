import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load stock data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    st.write("Columns in the data:", data.columns)
    st.write("First few rows of the data:", data.head())

    if 'Adj Close' in data.columns:
        close_column = 'Adj Close'
    elif 'Close' in data.columns:
        close_column = 'Close'
    else:
        st.error("Missing 'Adj Close' or 'Close' column.")
        return pd.DataFrame()

    data['Return'] = data[close_column].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data[close_column]).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data[close_column]).ema_indicator()
    data['Adj Close'] = data[close_column]  # Ensure consistency
    data.dropna(inplace=True)
    return data

# Prepare data
def prepare_data(data, lookback=14):
    features = ['Adj Close', 'Volume', 'RSI', 'EMA']
    features = [f for f in features if f in data.columns]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, lookback)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler, features

# Predict using uploaded model
def predict_with_model(model_file, X_test, scaler, features):
    model = load_model(model_file)
    y_pred = model.predict(X_test)
    dummy_features = np.zeros((len(y_pred), len(features) - 1))
    rescaled = scaler.inverse_transform(np.concatenate([y_pred, dummy_features], axis=1))
    return rescaled[:, 0]

# Streamlit UI
st.title("Stock Price Prediction with LSTM")
stock_symbol = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
model_file = st.file_uploader("Upload your LSTM model (.h5)", type=["h5"])

if stock_symbol:
    data = load_stock_data(stock_symbol, start_date, end_date)
    if data.empty:
        st.stop()

    X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)

    if model_file:
        y_pred_rescaled = predict_with_model(model_file, X_test, scaler, features)

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label="True Prices", alpha=0.7)
        plt.plot(y_pred_rescaled, label="Predicted Prices", alpha=0.7)
        plt.title(f"True vs Predicted Prices for {stock_symbol}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)

        # Show last 5 predictions
        st.subheader("Last 5 Predicted Prices:")
        st.write(y_pred_rescaled[-5:])
    else:
        st.info("Please upload a model to start prediction.")
