import streamlit as st
import yfinance as yf
from keras.models import load_model
import numpy as np
import pandas as pd

st.title("📈 Stock Predictor")

ticker = st.text_input("Enter Stock Ticker", "XUS.TO")

if st.button("Predict"):
    df = yf.download(ticker, start="2018-01-01")
    df["10_MA"] = df["Close"].rolling(window=10).mean()
    df["50_MA"] = df["Close"].rolling(window=50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    features = df[["Open", "Close", "10_MA", "50_MA", "RSI"]].values[-60:]
    features = features.reshape(1, 60, 5)

    model = load_model("xus_model_v2.h5", compile=False)
    prediction = model.predict(features)[0][0]

    st.success(f"Predicted Next Price: ${prediction:.2f}")
