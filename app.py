import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from nlp_module import extract_metric, sentiment_score, detect_anomalies
from forecast_module import get_forecast
from strategy_module import recommend_action

st.set_page_config(page_title="FinDocGPT", layout="wide")

st.title("📊 FinDocGPT – AI for Financial Insights & Strategy")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Earnings Report (txt)", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        revenue = extract_metric(text, "What is the company's revenue for the quarter?")
        profit = extract_metric(text, "What is the company's net profit?")
        sentiment = sentiment_score(text)
        
        st.subheader("📄 Document Insights")
        st.write(f"**Revenue:** {revenue}")
        st.write(f"**Net Profit:** {profit}")
        st.write(f"**Sentiment Score:** {sentiment}")

with col2:
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    if ticker:
        df, forecast = get_forecast(ticker)
        current_price = df["y"].iloc[-1]
        action, change = recommend_action(sentiment if uploaded_file else 0, forecast, current_price)

        st.subheader("📈 Price Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Investment Recommendation")
        st.write(f"**Action:** {action}")
        st.write(f"**Expected Change:** {change:.2%}")
