import argparse

import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

from compare_models import run_lstm
from preprocess import prepare_time_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LSTM for time-series forecasting (PyTorch).")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="5y", help="History period, e.g., 5y")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seq-len", type=int, default=30, help="LSTM lookback window size")
    parser.add_argument("--hidden-size", type=int, default=32, help="LSTM hidden size")
    parser.add_argument("--epochs", type=int, default=40, help="LSTM training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="LSTM learning rate")
    args = parser.parse_args()

    raw = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False).reset_index()
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    df = prepare_time_series(raw, date_col="Date", target_col="Close")
    series = df["Close"].dropna()
    split_idx = int(len(series) * (1 - args.test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    metrics, preds = run_lstm(
        train=train,
        test=test,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    mae = mean_absolute_error(test.values, preds.values)
    rmse = mean_squared_error(test.values, preds.values) ** 0.5

    print(f"Ticker: {args.ticker}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(
        f"LSTM params: seq_len={args.seq_len}, hidden_size={args.hidden_size}, "
        f"epochs={args.epochs}, lr={args.lr}"
    )
    print(f"LSTM MAE: {mae:.4f}")
    print(f"LSTM RMSE: {rmse:.4f}")
    print(f"Model name: {metrics['model']}")


if __name__ == "__main__":
    main()
