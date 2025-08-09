import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# --- STAGE 1: Insights & Analysis ---

def load_and_extract_text(file):
    # Person A: Implement file reading (.txt/.pdf)
    return "Sample extracted text from uploaded report."

def perform_qa(document_text, question):
    # Person A: Replace with OpenAI Q&A call
    return "Sample answer to: " + question

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

    # Stage 1: Upload + Q&A + Sentiment
    st.header("Stage 1: Document Insights")
    uploaded_file = st.file_uploader("Upload Earnings Report (.txt)", type=["txt"])
    document_text = ""
    if uploaded_file:
        document_text = load_and_extract_text(uploaded_file)
        st.write("Extracted Text Preview:", document_text[:300] + "...")
        revenue = perform_qa(document_text, "What is the company's revenue for the quarter?")
        profit = perform_qa(document_text, "What is the company's net profit?")
        sentiment = analyze_sentiment(document_text)
        st.write(f"**Revenue:** {revenue}")
        st.write(f"**Net Profit:** {profit}")
        st.write(f"**Sentiment Score:** {sentiment}")

    else:
        sentiment = 0

    # Stage 2: Forecasting
    st.header("Stage 2: Price Forecast")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")
    if ticker:
        df = fetch_stock_data(ticker)
        forecast = forecast_prices(df)
        current_price = df["y"].iloc[-1]
        render_price_chart(df, forecast)

        # Stage 3: Recommendation
        st.header("Stage 3: Investment Recommendation")
        action, change = generate_recommendation(sentiment, forecast, current_price)
        change_pct = change * 100
        st.write(f"**Action:** {action}")
        st.write(f"**Expected Price Change:** {change_pct:.2f}%")

    # Stage 4 & 5: Integration and polish to come

if __name__ == "__main__":
    main()
