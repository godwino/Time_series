import argparse
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import prepare_time_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prophet for time-series forecasting.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="5y", help="History period, e.g., 5y")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet is not installed. Run: pip install prophet") from exc

    raw = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False).reset_index()
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df = prepare_time_series(raw, date_col="Date", target_col="Close").reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})

    split_idx = int(len(df) * (1 - args.test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    future = test[["ds"]].copy()
    forecast = model.predict(future)
    preds = forecast["yhat"].values

    mae = mean_absolute_error(test["y"].values, preds)
    rmse = mean_squared_error(test["y"].values, preds) ** 0.5

    print(f"Ticker: {args.ticker}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(f"Prophet MAE: {mae:.4f}")
    print(f"Prophet RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
