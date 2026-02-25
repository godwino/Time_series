import argparse
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import prepare_time_series


def naive_forecast(train: pd.Series, test_len: int) -> pd.Series:
    """Forecast every future point as the last observed training value."""
    return pd.Series([train.iloc[-1]] * test_len)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a naive time-series baseline.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="5y", help="History period, e.g., 5y")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    raw = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False).reset_index()
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df = prepare_time_series(raw, date_col="Date", target_col="Close")
    series = df["Close"].dropna()

    split_idx = int(len(series) * (1 - args.test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    preds = naive_forecast(train, len(test))
    mae = mean_absolute_error(test.values, preds.values)
    rmse = mean_squared_error(test.values, preds.values) ** 0.5

    print(f"Ticker: {args.ticker}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(f"Naive MAE: {mae:.4f}")
    print(f"Naive RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
