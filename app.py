import streamlit as st
import os
from streamlit.errors import StreamlitSecretNotFoundError
import pandas as pd
import numpy as np
import re
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from textblob import TextBlob
from openai import OpenAI
from datetime import timedelta
import plotly.graph_objects as go
from file_utils import read_uploaded_file, chunk_text_semantic, chunk_text_fixed
from qa_utils import choose_best_chunk, openai_answer_with_context, perform_qa
import smtplib
from email.mime.text import MIMEText

st.set_page_config(page_title="FinDocGPT (Hackathon)", layout="wide")

# Load API key safely
api_key = os.getenv("OPENAI_API_KEY")
try:
    if not api_key:
        api_key = st.secrets["OPENAI_API_KEY"]
except StreamlitSecretNotFoundError:
    api_key = "sk-5678mnopqrstuvwx5678mnopqrstuvwx5678mnop"  # fallback key

client = OpenAI(api_key=api_key) if api_key else None

# Helper functions (chunking, sentiment, anomalies, forecasting, recommendations, visualization, alerts)
def chunk_text(text, chunk_size=1000, overlap=200):
    return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap)

def sentiment_textblob(text):
    if not text or len(text.strip()) < 10:
        return 0.0, "neutral", "TextBlob"
    p = float(TextBlob(text).sentiment.polarity)
    label = "positive" if p > 0.05 else ("negative" if p < -0.05 else "neutral")
    return p, label, "TextBlob"

def detect_anomalies_from_text(text):
    anomalies = []
    patterns = {
        "Revenue": r"Revenue[\s:\-–]*\$?([\d,]+\.?\d*)",
        "Net Income": r"Net Income[\s:\-–]*\$?([\d,]+\.?\d*)"
    }
    for name, patt in patterns.items():
        vals = re.findall(patt, text, flags=re.IGNORECASE)
        if len(vals) >= 2:
            try:
                last = float(vals[-1].replace(",", ""))
                prev = float(vals[-2].replace(",", ""))
                if prev != 0:
                    change = (last - prev) / abs(prev)
                    if abs(change) > 0.15:
                        anomalies.append({
                            "name": f"{name.lower().replace(' ','_')}_qoq",
                            "detail": f"{name} {change*100:+.1f}% QoQ",
                            "severity": "high" if abs(change) > 0.30 else "medium"
                        })
            except Exception:
                continue
    return anomalies

def fetch_stock_data(ticker):
    try:
        raw = yf.download(ticker, period="2y", interval="1d", progress=False)
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return None
    if raw is None or raw.empty:
        st.error(f"No price data for ticker: {ticker}")
        return None

    df = raw.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    close_col = None
    for col in df.columns:
        if col.lower() == "close":
            close_col = col
            break
        if ticker.lower() in col.lower() and "close" in col.lower():
            close_col = col
            break
    if close_col is None:
        st.error("Could not find 'Close' price column in data.")
        return None

    df = df.rename(columns={"Date": "ds", close_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if df.empty:
        st.error("No valid price rows after cleaning.")
        return None
    return df

def forecast_prices(df, periods=30, min_history=60):
    if df is None or df.empty:
        return None, None, None
    df = df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    if len(df) < min_history:
        last_price = float(df["y"].iloc[-1])
        future_dates = pd.date_range(start=df["ds"].iloc[-1] + timedelta(days=1), periods=periods)
        forecast = pd.DataFrame({
            "ds": pd.concat([df["ds"], pd.Series(future_dates)]).reset_index(drop=True),
            "yhat": pd.concat([df["y"], pd.Series([last_price]*periods)]).reset_index(drop=True)
        })
        return forecast, None, None
    try:
        train = df.iloc[:-periods].reset_index(drop=True)
        test = df.iloc[-periods:].reset_index(drop=True)
        model = Prophet()
        model.fit(train)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        try:
            preds = forecast.set_index("ds").loc[test["ds"], "yhat"].values
            mae = mean_absolute_error(test["y"].values, preds)
            rmse = mean_squared_error(test["y"].values, preds, squared=False)
        except Exception:
            mae, rmse = None, None
        return forecast, mae, rmse
    except Exception as e:
        st.error(f"Prophet training/predict error: {e}")
        return None, None, None

def generate_recommendation(sentiment_score, forecast, current_price, anomalies):
    risk_notes = [a["detail"] for a in anomalies] if anomalies else []
    if forecast is None or forecast.empty or current_price is None:
        trace = f"Insufficient forecast/price. Sent={sentiment_score:+.2f}"
        return "HOLD", 0.0, trace, risk_notes
    try:
        last_yhat = float(forecast["yhat"].iloc[-1])
    except Exception:
        return "HOLD", 0.0, "Forecast yhat unavailable", risk_notes
    change = (last_yhat - current_price) / current_price
    if sentiment_score > 0.2 and change > 0.05:
        action = "BUY"
    elif sentiment_score < -0.2 and change < -0.05:
        action = "SELL"
    else:
        action = "HOLD"
    trace = f"Trend={'Up' if change > 0 else 'Down'}, Sent={sentiment_score:+.2f} → {action} (conf={abs(change):.3f})"
    return action, change, trace, risk_notes

def render_price_chart(df, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Actual Price"))
    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
    fig.update_layout(title="Stock Price & Forecast", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

def send_email_alert(to_email, subject, body, smtp_server="smtp.gmail.com", smtp_port=587,
                     from_email=None, from_password=None):
    if not from_email or not from_password:
        st.warning("Email credentials not set, skipping email alert.")
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
        st.success(f"Alert email sent to {to_email}")
    except Exception as e:
        st.error(f"Error sending email: {e}")

def check_alerts(watchlist, forecasts, current_prices):
    alerts = []
    for ticker, data in watchlist.items():
        threshold = data.get("threshold", 0.05)
        forecast = forecasts.get(ticker)
        current_price = current_prices.get(ticker)
        if forecast is None or current_price is None:
            continue
        try:
            last_yhat = float(forecast["yhat"].iloc[-1])
            change = (last_yhat - current_price) / current_price
            if abs(change) >= threshold:
                alerts.append({
                    "ticker": ticker,
                    "change_pct": change * 100,
                    "threshold": threshold * 100,
                    "message": f"{ticker} forecast change {change*100:+.2f}% exceeds threshold {threshold*100:.2f}%"
                })
        except Exception:
            continue
    return alerts

def generate_summary_openai(text, model="gpt-4o-mini", max_tokens=300):
    if client is None:
        return "OpenAI API key not set or client not initialized."
    prompt = (
        "You are a helpful financial analyst assistant. "
        "Summarize the following financial report text concisely, focusing on key highlights, "
        "financial performance, and outlook:\n\n"
        + text[:4000]
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You summarize financial documents concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"

# Initialize session state variables (document text, chunks, sentiment, anomalies, etc.)
if "document_text" not in st.session_state:
    st.session_state.document_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "qa_results" not in st.session_state:
    st.session_state.qa_results = []
if "sentiment" not in st.session_state:
    st.session_state.sentiment = {"score": 0.0, "label": "neutral", "model": "TextBlob"}
if "anomalies" not in st.session_state:
    st.session_state.anomalies = []
if "df" not in st.session_state:
    st.session_state.df = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None
if "mae" not in st.session_state:
    st.session_state.mae = None
if "rmse" not in st.session_state:
    st.session_state.rmse = None
if "current_price" not in st.session_state:
    st.session_state.current_price = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = {}
if "summary" not in st.session_state:
    st.session_state.summary = ""

if "watchlist" not in st.session_state:
    st.session_state.watchlist = {}
if "watchlist_forecasts" not in st.session_state:
    st.session_state.watchlist_forecasts = {}
if "watchlist_prices" not in st.session_state:
    st.session_state.watchlist_prices = {}
if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()
if "alert_email" not in st.session_state:
    st.session_state.alert_email = ""

# Sidebar controls
st.sidebar.title("FinDocGPT Controls")
page = st.sidebar.radio("Navigate", ["Upload & Insights", "Forecast", "Recommendation", "Metrics", "Watchlist & Alerts"])
st.sidebar.markdown("---")

st.sidebar.subheader("Forecast Controls")
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NFLX", "META"]
ticker = st.sidebar.selectbox("Select Ticker", options=tickers, index=0)
model_choice = st.sidebar.selectbox("OpenAI Model", options=["gpt-4o-mini", "gpt-3.5-turbo"], index=0)
run_forecast = st.sidebar.button("Run Forecast")

st.sidebar.markdown("---")
st.sidebar.subheader("Alert Email")
email_input = st.sidebar.text_input("Enter your email for alerts", value=st.session_state.alert_email)
if email_input != st.session_state.alert_email:
    st.session_state.alert_email = email_input

# Run forecast action
if run_forecast and ticker:
    with st.spinner("Fetching & forecasting..."):
        st.session_state.df = fetch_stock_data(ticker)
        if st.session_state.df is not None:
            st.session_state.forecast, st.session_state.mae, st.session_state.rmse = forecast_prices(
                st.session_state.df, periods=30, min_history=60
            )
            st.session_state.current_price = float(st.session_state.df["y"].iloc[-1]) if not st.session_state.df.empty else None

            if ticker in st.session_state.watchlist:
                st.session_state.watchlist_forecasts[ticker] = st.session_state.forecast
                st.session_state.watchlist_prices[ticker] = st.session_state.current_price

# Pages
if page == "Upload & Insights":
    st.header("Stage 1 — Upload & Insights")
    uploaded = st.file_uploader("Upload report (.txt or .pdf)", type=["txt", "pdf"])
    if uploaded:
        txt = read_uploaded_file(uploaded)
        st.session_state.document_text = txt
        st.session_state.chunks = chunk_text(txt)
        score_tb, label_tb, model_tb = sentiment_textblob(txt)
        st.session_state.sentiment = {"score": score_tb, "label": label_tb, "model": model_tb}
        st.session_state.anomalies = detect_anomalies_from_text(txt)

        with st.spinner("Generating document summary..."):
            summary = generate_summary_openai(txt, model=model_choice)
            st.session_state.summary = summary

        st.subheader("Document preview")
        st.write(txt[:1000] + ("..." if len(txt) > 1000 else ""))

        st.subheader("Document Summary")
        if st.session_state.summary:
            st.write(st.session_state.summary)
        else:
            st.info("No summary generated.")

        st.subheader("Sentiment")
        st.metric("Sentiment score", f"{score_tb:+.3f}", label_tb)
        st.markdown(f"*Model used: {model_tb}*")

        st.subheader("Detected anomalies (text heuristic)")
        if st.session_state.anomalies:
            for a in st.session_state.anomalies:
                st.warning(f"{a['detail']} (severity: {a['severity']})")
        else:
            st.success("No simple anomalies flagged.")

        st.subheader("Quick Q&A (sourced)")
        default_questions = [
            "What is the company's revenue?",
            "What is the company's net income?",
            "What did management say about outlook?"
        ]
        st.session_state.qa_results = []
        use_openai = st.sidebar.checkbox("Use OpenAI for Q&A (RAG)", value=False)
        for q in default_questions:
            if use_openai and client is not None:
                ans, snip = openai_answer_with_context(q, st.session_state.chunks, n_chunks=2, model=model_choice)
            else:
                ans, snip = perform_qa(txt, q)
            st.session_state.qa_results.append({"question": q, "answer": ans, "source": snip})
            st.markdown(f"**Q:** {q}")
            st.info(f"A: {ans}")
            st.caption(f"Source snippet: {snip}")

elif page == "Forecast":
    st.header("Stage 2 — Forecast")
    st.write("Use the sidebar to select ticker and click 'Run Forecast'.")
    if st.session_state.df is not None:
        render_price_chart(st.session_state.df, st.session_state.forecast)
        if st.session_state.mae is not None:
            st.write(f"MAE: {st.session_state.mae:.4f}")
            if st.session_state.rmse is not None:
                st.write(f"RMSE: {st.session_state.rmse:.4f}")
            else:
                st.write("RMSE: N/A (fallback or error)")
        else:
            st.info("Forecast metrics not available (fallback or insufficient history).")
    else:
        st.info("No forecast data yet. Select ticker and run forecast from the sidebar.")

elif page == "Recommendation":
    st.header("Stage 3 — Recommendation")
    if not st.session_state.document_text:
        st.info("Upload a document on the 'Upload & Insights' page first.")
    elif st.session_state.forecast is None:
        st.info("Run a forecast from the sidebar first.")
    else:
        action, change, trace, risks = generate_recommendation(
            st.session_state.sentiment["score"],
            st.session_state.forecast,
            st.session_state.current_price,
            st.session_state.anomalies
        )
        st.subheader(f"Recommendation: {action}")
        st.write(f"Expected change: {change*100:+.2f}%")
        st.markdown("**Reasoning:**")
        st.write(trace)
        if risks:
            st.markdown("**Risks/Anomalies detected:**")
            for r in risks:
                st.warning(r)

elif page == "Metrics":
    st.header("Stage 4 — Metrics & Benchmark")
    if st.session_state.mae is not None and st.session_state.rmse is not None:
        st.metric("Forecast MAE", f"{st.session_state.mae:.4f}")
        st.metric("Forecast RMSE", f"{st.session_state.rmse:.4f}")
    else:
        st.info("No forecast metrics available yet.")

    st.subheader("Q&A Sample")
    sample_qs = [
        "What is the company's revenue?",
        "What is the company's net income?",
        "What did management say about outlook?"
    ]
    for q in sample_qs:
        if client:
            ans, _ = openai_answer_with_context(q, st.session_state.chunks, n_chunks=2, model=model_choice)
        else:
            ans, _ = perform_qa(st.session_state.document_text, q)
        st.markdown(f"**Q:** {q}")
        st.info(f"A: {ans}")

elif page == "Watchlist & Alerts":
    st.header("Stage 5 — Watchlist & Alerts")
    st.write("Manage your watchlist and receive alerts based on forecast thresholds.")

    # Add ticker to watchlist
    new_ticker = st.text_input("Add ticker to watchlist")
    threshold = st.slider("Alert threshold (%)", min_value=1, max_value=20, value=5, step=1) / 100.0
    if st.button("Add to Watchlist") and new_ticker:
        st.session_state.watchlist[new_ticker.upper()] = {"threshold": threshold}
        st.success(f"Added {new_ticker.upper()} with threshold {threshold*100:.1f}%")

    if st.session_state.watchlist:
        st.subheader("Current Watchlist")
        for tck, data in st.session_state.watchlist.items():
            st.write(f"- {tck} with alert threshold {data['threshold']*100:.1f}%")

    alerts = check_alerts(st.session_state.watchlist, st.session_state.watchlist_forecasts, st.session_state.watchlist_prices)
    if alerts:
        st.subheader("Alerts")
        for alert in alerts:
            st.warning(alert["message"])
            if st.session_state.alert_email and alert["ticker"] not in st.session_state.sent_alerts:
                send_email_alert(
                    st.session_state.alert_email,
                    f"Alert for {alert['ticker']}",
                    alert["message"],
                    from_email=st.secrets.get("email_address"),
                    from_password=st.secrets.get("email_password")
                )
                st.session_state.sent_alerts.add(alert["ticker"])
    else:
        st.info("No alerts at this time.")

