import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from preprocess import prepare_time_series


def naive_forecast(train: pd.Series, test_len: int) -> pd.Series:
    # Baseline: repeat the last known value for all future timestamps.
    return pd.Series([train.iloc[-1]] * test_len)


def run_naive(train: pd.Series, test: pd.Series) -> dict:
    preds = naive_forecast(train, len(test))
    metrics = {
        "model": "Naive",
        "mae": mean_absolute_error(test.values, preds.values),
        "rmse": mean_squared_error(test.values, preds.values) ** 0.5,
    }
    preds.index = test.index
    return metrics, preds


def run_arima(train: pd.Series, test: pd.Series, order) -> dict:
    # Fit ARIMA on train only, then forecast exactly the test horizon.
    fitted = ARIMA(train, order=order).fit()
    preds = pd.Series(fitted.forecast(steps=len(test)), index=test.index)
    metrics = {
        "model": f"ARIMA{order}",
        "mae": mean_absolute_error(test.values, preds.values),
        "rmse": mean_squared_error(test.values, preds.values) ** 0.5,
    }
    return metrics, preds


def run_prophet(df: pd.DataFrame, split_idx: int) -> dict:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Prophet is not installed. Run: pip install prophet") from exc

    # Prophet requires columns named ds (datetime) and y (target).
    prophet_df = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    train = prophet_df.iloc[:split_idx].copy()
    test = prophet_df.iloc[split_idx:].copy()

    # Keep seasonality simple and interpretable for this portfolio baseline.
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train)
    preds = pd.Series(model.predict(test[["ds"]])["yhat"].values, index=pd.to_datetime(test["ds"]))

    metrics = {
        "model": "Prophet",
        "mae": mean_absolute_error(test["y"].values, preds.values),
        "rmse": mean_squared_error(test["y"].values, preds.values) ** 0.5,
    }
    return metrics, preds


def save_comparison_plot(test: pd.Series, model_preds: dict, plot_path: str) -> None:
    # Plot actual values against each model's predictions in a presentation-ready style.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(13, 7))
    plt.plot(test.index, test.values, label="Actual", linewidth=2.5, color="#1f2937")
    palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    color_idx = 0
    for model_name, preds in model_preds.items():
        plt.plot(
            preds.index,
            preds.values,
            label=model_name,
            alpha=0.95,
            linewidth=1.8,
            color=palette[color_idx % len(palette)],
        )
        color_idx += 1

    plt.title("Forecast Comparison: Actual vs Predicted", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_predictions_frame(test: pd.Series, model_preds: dict) -> pd.DataFrame:
    frame = pd.DataFrame({"Date": test.index, "Actual": test.values})
    for model_name, preds in model_preds.items():
        frame[model_name] = preds.reindex(test.index).values
    return frame


def add_business_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    # Translate raw errors into portfolio-friendly comparison indicators.
    out = results_df.copy()
    out["rank"] = np.arange(1, len(out) + 1)
    best_mae = out["mae"].iloc[0]
    out["mae_vs_best_pct"] = ((out["mae"] / best_mae) - 1) * 100
    return out


def save_exec_summary(
    summary_path: str,
    ticker: str,
    period: str,
    train_size: int,
    test_size: int,
    ranked_df: pd.DataFrame,
    plot_path: str,
    metrics_path: str,
    predictions_path: str,
) -> None:
    winner = ranked_df.iloc[0]
    lines = [
        "# Executive Summary",
        "",
        "## Forecasting Portfolio Snapshot",
        f"- Asset: `{ticker}`",
        f"- History window: `{period}`",
        f"- Train points: `{train_size}`",
        f"- Test points: `{test_size}`",
        "",
        "## Best Model",
        f"- Winner: `{winner['model']}`",
        f"- MAE: `{winner['mae']:.4f}`",
        f"- RMSE: `{winner['rmse']:.4f}`",
        "",
        "## Generated Artifacts",
        f"- Metrics table: `{metrics_path}`",
        f"- Predictions table: `{predictions_path}`",
        f"- Comparison chart: `{plot_path}`",
        "",
        "## Business Interpretation",
        "- The selected model minimizes forecast error on held-out data.",
        "- This supports stronger short-term planning and risk-aware decisions.",
        "- Retraining cadence should be set (for example weekly/monthly) based on use case volatility.",
    ]
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Naive, ARIMA, and Prophet models.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", type=str, default="5y", help="History period, e.g., 5y")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--p", type=int, default=5, help="ARIMA p")
    parser.add_argument("--d", type=int, default=1, help="ARIMA d")
    parser.add_argument("--q", type=int, default=0, help="ARIMA q")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="outputs/forecast_comparison.png",
        help="Where to save actual-vs-predicted comparison plot",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="outputs/model_metrics.csv",
        help="Where to save model metrics table",
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default="outputs/test_predictions.csv",
        help="Where to save actual/predicted test values",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="outputs/executive_summary.md",
        help="Where to save executive summary markdown",
    )
    args = parser.parse_args()

    # Pull adjusted historical prices from Yahoo Finance.
    raw = yf.download(args.ticker, period=args.period, auto_adjust=True, progress=False).reset_index()
    if raw.empty:
        raise ValueError("No data returned from Yahoo Finance.")

    # Shared preprocessing: date parsing, daily frequency, missing-value fill.
    df = prepare_time_series(raw, date_col="Date", target_col="Close")
    series = df["Close"].dropna()
    # Time-based split (never random split for forecasting tasks).
    split_idx = int(len(series) * (1 - args.test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    # Evaluate all candidate models on the same split for fair comparison.
    results = []
    model_preds = {}

    naive_metrics, naive_preds = run_naive(train, test)
    results.append(naive_metrics)
    model_preds[naive_metrics["model"]] = naive_preds

    arima_metrics, arima_preds = run_arima(train, test, order=(args.p, args.d, args.q))
    results.append(arima_metrics)
    model_preds[arima_metrics["model"]] = arima_preds

    prophet_metrics, prophet_preds = run_prophet(df, split_idx=split_idx)
    results.append(prophet_metrics)
    model_preds[prophet_metrics["model"]] = prophet_preds

    # Rank models by error (lower MAE/RMSE is better), then enrich for business-facing reporting.
    results_df = pd.DataFrame(results).sort_values(["mae", "rmse"], ascending=True).reset_index(drop=True)
    ranked_df = add_business_columns(results_df)
    predictions_df = build_predictions_frame(test, model_preds)

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(metrics_path, index=False)

    predictions_path = Path(args.predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)

    save_comparison_plot(test, model_preds, args.plot_path)
    save_exec_summary(
        summary_path=args.summary_path,
        ticker=args.ticker,
        period=args.period,
        train_size=len(train),
        test_size=len(test),
        ranked_df=ranked_df,
        plot_path=args.plot_path,
        metrics_path=args.metrics_path,
        predictions_path=args.predictions_path,
    )

    print(f"Ticker: {args.ticker}")
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("\nModel ranking (lower is better):")
    print(
        ranked_df.to_string(
            index=False,
            formatters={
                "mae": "{:.4f}".format,
                "rmse": "{:.4f}".format,
                "mae_vs_best_pct": "{:.2f}".format,
            },
        )
    )
    print(f"\nSaved plot: {Path(args.plot_path)}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved summary: {Path(args.summary_path)}")


if __name__ == "__main__":
    main()
