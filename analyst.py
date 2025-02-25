import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import date
import os

token_name = os.getenv("TOKEN_NAME")
token_pair = os.getenv("TOKEN_PAIR")
token_data = yf.download(token_pair, start='2022-01-01', end=date.today(), interval='1d')

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(token_data[['Close']])

lookback = 30
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, lookback)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(12, 6))
plt.plot(token_data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Price')
plt.plot(token_data.index[-len(y_test):], predictions, label='Predicted Price')
plt.legend()
plt.title(f"{token_name} Price Prediction using LSTM")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()