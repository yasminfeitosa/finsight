# alerts_page.py

import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

# ----------- Helpers ------------

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
    if "Date" not in df.columns or "Close" not in df.columns:
        st.error("Unexpected data format from yfinance")
        return None

    ds = pd.to_datetime(df["Date"], errors="coerce")
    y = pd.to_numeric(df["Close"], errors="coerce")
    df_clean = pd.DataFrame({"ds": ds, "y": y})
    df_clean = df_clean.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    return df_clean

def forecast_prices(df, periods=7, min_history=60):
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
            "yhat": pd.concat([df["y"], pd.Series([last_price] * periods)]).reset_index(drop=True)
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
        preds = forecast.set_index("ds").loc[test["ds"], "yhat"].values
        mae = mean_absolute_error(test["y"].values, preds)
        rmse = mean_squared_error(test["y"].values, preds, squared=False)
        return forecast, mae, rmse
    except Exception as e:
        st.error(f"Prophet error: {e}")
        return None, None, None

def check_alerts(watchlist, thresholds, current_prices, forecasts):
    alerts = []
    for ticker in watchlist:
        price = current_prices.get(ticker)
        forecast = forecasts.get(ticker)
        if price is None or forecast is None:
            continue
        try:
            last_forecast_price = float(forecast["yhat"].iloc[-1])
            change = (last_forecast_price - price) / price
            threshold = thresholds.get(ticker, 0.05)  # default 5%
            if abs(change) >= threshold:
                alerts.append({
                    "ticker": ticker,
                    "current_price": price,
                    "forecast_price": last_forecast_price,
                    "change": change,
                    "threshold": threshold
                })
        except Exception:
            continue
    return alerts

def send_email_alert(receiver_email, subject, body, sender_email, sender_password, smtp_server="smtp.gmail.com", smtp_port=587):
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ----------- Streamlit UI -----------

st.set_page_config(page_title="FinDocGPT - Watchlist & Alerts", layout="wide")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "thresholds" not in st.session_state:
    st.session_state.thresholds = {}
if "watchlist_forecasts" not in st.session_state:
    st.session_state.watchlist_forecasts = {}
if "watchlist_prices" not in st.session_state:
    st.session_state.watchlist_prices = {}

st.title("Customization & Alerts")

col1, col2 = st.columns([3,1])
with col1:
    new_ticker = st.text_input("Add ticker to watchlist", "").upper()
with col2:
    if st.button("Add ticker"):
        if new_ticker and re.fullmatch(r"[A-Z0-9\-.]{1,10}", new_ticker):
            if new_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker)
                st.success(f"Added {new_ticker} to watchlist")
            else:
                st.warning(f"{new_ticker} already in watchlist")
        else:
            st.error("Invalid ticker format")

if st.session_state.watchlist:
    st.subheader("Watchlist and Thresholds")
    for t in st.session_state.watchlist:
        threshold = st.number_input(f"Alert threshold for {t} (fraction, e.g. 0.05 = 5%)", min_value=0.0, max_value=1.0, value=st.session_state.thresholds.get(t, 0.05), key=f"thresh_{t}")
        st.session_state.thresholds[t] = threshold

    st.subheader("Email Alerts")
    email = st.text_input("Enter your email address to receive alerts")
    sender_email = st.text_input("Sender email (Gmail recommended for SMTP)")
    sender_password = st.text_input("Sender email password or app password", type="password")

    if st.button("Check Alerts Now"):
        if not st.session_state.watchlist:
            st.error("Add tickers to your watchlist first.")
        elif not email:
            st.error("Please enter your email to receive alerts.")
        elif not sender_email or not sender_password:
            st.error("Please enter sender email credentials to send alerts.")
        else:
            current_prices = {}
            forecasts = {}

            for t in st.session_state.watchlist:
                dfw = fetch_stock_data(t)
                if dfw is not None and not dfw.empty:
                    current_prices[t] = float(dfw["y"].iloc[-1])
                    fc, _, _ = forecast_prices(dfw, periods=7)
                    forecasts[t] = fc

            st.session_state.watchlist_prices = current_prices
            st.session_state.watchlist_forecasts = forecasts

            alerts = check_alerts(st.session_state.watchlist, st.session_state.thresholds, current_prices, forecasts)
            if alerts:
                st.error("Alerts triggered:")
                for a in alerts:
                    chg_pct = a["change"] * 100
                    st.write(f"Ticker {a['ticker']} forecast change {chg_pct:+.2f}% exceeds threshold {a['threshold']*100:.2f}%. Current price: {a['current_price']:.2f}, Forecast price: {a['forecast_price']:.2f}")

                body = "Alert Summary for your watchlist:\n\n"
                for a in alerts:
                    chg_pct = a["change"] * 100
                    body += f"Ticker {a['ticker']} forecast change {chg_pct:+.2f}% exceeds threshold {a['threshold']*100:.2f}%.\nCurrent price: {a['current_price']:.2f}\nForecast price: {a['forecast_price']:.2f}\n\n"

                subject = "FinDocGPT Watchlist Alerts"
                if send_email_alert(email, subject, body, sender_email, sender_password):
                    st.success("Alert email sent successfully!")
                else:
                    st.error("Failed to send alert email.")
            else:
                st.success("No alerts triggered.")
else:
    st.info("Add tickers to your watchlist to get started.")
