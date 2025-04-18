import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from sklearn.preprocessing import MinMaxScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/mnt/data/NSEI 2015-2023.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Stock Market Analysis")
option = st.sidebar.selectbox("Select an Option", ["Data Overview", "Stock Price Prediction"])

if option == "Data Overview":
    st.title("Stock Market Data Overview")
    st.write(df.head())
    
    # Plot historical stock prices
    st.subheader("Stock Price Over Time")
    fig = px.line(df, x=df.index, y='Close', title='Stock Closing Price')
    st.plotly_chart(fig)
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    fig2 = px.line(df, x=df.index, y=['Close', 'MA50', 'MA200'], title='Moving Averages')
    st.plotly_chart(fig2)
    
elif option == "Stock Price Prediction":
    st.title("Stock Price Prediction using LSTM")
    
    # Data Preprocessing
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences for training
    def create_sequences(data, step=60):
        X, y = [], []
        for i in range(len(data) - step):
            X.append(data[i:i+step])
            y.append(data[i+step])
        return np.array(X), np.array(y)
    
    X, y = create_sequences(data_scaled)
    X_train, X_test, y_train, y_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):], y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    
    # Build LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model (for simplicity, using a small number of epochs)
    model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_test, y_test))
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))
    
    # Plot predictions vs actual values
    st.subheader("Predicted vs Actual Stock Prices")
    fig3 = px.line(title='Stock Price Prediction')
    fig3.add_scatter(x=df.index[-len(y_test_actual):], y=y_test_actual.flatten(), mode='lines', name='Actual')
    fig3.add_scatter(x=df.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted')
    st.plotly_chart(fig3)
    
    st.write("Note: This is a basic LSTM model trained for demonstration purposes. For better accuracy, tune hyperparameters and increase epochs.")

st.sidebar.info("Developed by Sanskar Vaibhav ")
