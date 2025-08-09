import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from nlp_module import extract_metric

# --- STAGE 1: Insights & Analysis ---
@st.cache_data
def load_financebench_data():
    df = pd.read_json("docs/financebench_open_source.jsonl", lines=True)
    return df.to_string

def perform_qa(document_text, question):
    return "The answer is: " + extract_metric(document_text, question)

def analyze_sentiment(document_text):
    # Person A: Use TextBlob or similar
    return 0.1  # dummy positive sentiment

# --- STAGE 2: Financial Forecasting ---

def fetch_stock_data(ticker):
    # Person B: Use yfinance or API to get historical data
    dates = pd.date_range(start="2023-01-01", periods=100)
    prices = pd.Series(100 + (pd.np.random.randn(100).cumsum()), index=dates)
    df = pd.DataFrame({"ds": dates, "y": prices})
    return df

def forecast_prices(df):
    # Person B: Use Prophet or ARIMA to forecast prices
    forecast_dates = pd.date_range(start=df["ds"].iloc[-1], periods=30)
    forecast_prices = df["y"].iloc[-1] + pd.Series(range(30))  # dummy increasing forecast
    forecast = pd.DataFrame({"ds": forecast_dates, "yhat": forecast_prices})
    return forecast

# --- STAGE 3: Investment Strategy ---

def generate_recommendation(sentiment, forecast, current_price):
    # Person C: Implement decision logic based on sentiment and forecast
    future_price = forecast["yhat"].iloc[-1]
    change = (future_price - current_price) / current_price
    if sentiment > 0.2 and change > 0.05:
        return "BUY", change
    elif sentiment < -0.2 and change < -0.05:
        return "SELL", change
    else:
        return "HOLD", change

# --- STAGE 4: UI & Integration ---

def render_price_chart(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
def main():
    st.title("📊 FinDocGPT – Financial Document AI")

    # Stage 1: Data + Q&A + Sentiment
    st.header("Stage 1: Document Insights")

    df_reports = load_financebench_data()

    user_question = st.text_input("What do you want to know?")

    if st.button("Answer"):
        answer = perform_qa(df_reports, user_question)
        st.markdown("**Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()
