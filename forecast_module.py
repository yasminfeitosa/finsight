import yfinance as yf
from prophet import Prophet
import pandas as pd

def get_forecast(ticker, days_ahead=30):
    """Fetch data, train Prophet, and forecast."""
    df = yf.download(ticker, period="2y", interval="1d")
    df = df.reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    
    model = Prophet(daily_seasonality=True)
    model.fit(df[["ds", "y"]])
    
    future = model.make_future_dataframe(periods=days_ahead)
    forecast = model.predict(future)
    
    return df, forecast