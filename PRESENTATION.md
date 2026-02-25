# Time Series Forecasting Portfolio Presentation

## 1) Business Problem
Organizations need reliable forecasts to make better decisions on planning, budgeting, inventory, and risk management. Inaccurate forecasts create revenue loss and operational inefficiency.

## 2) Project Objective
Build a practical forecasting pipeline and benchmark multiple models to identify the most reliable predictor for future values.

## 3) Data Used
- Source: Yahoo Finance (via `yfinance`)
- Example asset: `AAPL`
- Target: daily adjusted `Close` price
- Window: configurable (default `5y`)

## 4) Methodology
1. Data ingestion and cleaning
2. Time index standardization (daily frequency)
3. Missing value handling (`ffill`/`bfill`)
4. Time-based train/test split
5. Model benchmarking:
   - Naive baseline
   - ARIMA (`p,d,q`)
   - Prophet
6. Evaluation metrics:
   - MAE
   - RMSE

## 5) Professional Outputs
After one command, the project generates:
- `outputs/forecast_comparison.png` (Actual vs Predicted chart)
- `outputs/model_metrics.csv` (ranked model metrics)
- `outputs/test_predictions.csv` (actual and predicted values)
- `outputs/executive_summary.md` (business-facing summary)

## 6) Key Result Framing
- The best model is selected using lowest MAE/RMSE on unseen test data.
- Results are transparent, reproducible, and easy to communicate to non-technical stakeholders.

## 7) Business Impact
- Improves planning confidence.
- Reduces forecasting risk.
- Supports repeatable, data-driven decision workflows.

## 8) Demo Command
```powershell
python src/compare_models.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0
```

## 9) Next-Phase Enhancements
- Add LSTM for deep learning comparison.
- Add walk-forward backtesting.
- Add automated hyperparameter search.
- Deploy as a lightweight dashboard (Streamlit).
