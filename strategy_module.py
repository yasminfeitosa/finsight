def recommend_action(sentiment, forecast_df, current_price, up_threshold=0.05, down_threshold=-0.05):
    """Return Buy/Sell/Hold decision."""
    future_price = forecast_df["yhat"].iloc[-1]
    price_change = (future_price - current_price) / current_price

    if sentiment > 0.3 and price_change > up_threshold:
        return "BUY", price_change
    elif sentiment < -0.3 and price_change < down_threshold:
        return "SELL", price_change
    else:
        return "HOLD", price_change