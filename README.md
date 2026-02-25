# Predictive Analytics With Time Series Data

## Portfolio Positioning
This project demonstrates production-style time-series forecasting with transparent evaluation, model comparison, and business-ready reporting outputs.

## Business Problem
Businesses need reliable forecasts to plan inventory, staffing, budgets, and risk. In finance and retail, poor forecasts lead to stockouts, overstock, missed revenue, and weak capital allocation.

## Project Goal
Build and compare time-series forecasting models to predict future values (for example: stock prices, sales, demand, or economic indicators) and identify the most robust approach across short- and medium-term horizons.

## Suggested Data Sources
- Yahoo Finance: historical stock prices and market indicators.
- Nasdaq Data Link: financial and macroeconomic datasets.
- UCI Machine Learning Repository: benchmark time-series datasets.

## Why Machine Learning
Classical methods are strong for linear, stationary series, but many real-world series are nonlinear and impacted by sudden shocks. ML and deep learning models can learn complex patterns and adapt better when engineered and validated correctly.

## Scope
- Time-series preprocessing and quality checks.
- Trend and seasonality decomposition.
- Feature engineering (lags, rolling statistics, calendar features).
- Model training and hyperparameter tuning.
- Backtesting and forecast evaluation.
- Business-facing interpretation and recommendations.

## Workflow
1. Define the target variable and forecast horizon.
2. Ingest and clean data (missing values, duplicates, time index alignment).
3. Perform exploratory time-series analysis.
4. Test stationarity (ADF), apply differencing if needed.
5. Build baseline and advanced models:
   - Baseline: Naive/Moving Average
   - Statistical: ARIMA/SARIMA
   - ML/DL: Prophet, LSTM
6. Tune hyperparameters with time-series cross-validation.
7. Compare models using MAE, RMSE, and MAPE.
8. Select the best model and generate future forecasts.
9. Summarize business impact and deployment considerations.

## Preprocessing and Modeling Tips
- Missing values: use forward-fill/backward-fill only after checking data meaning.
- Stationarity: validate with Dickey-Fuller before ARIMA-family models.
- Features: add lag variables and rolling windows carefully to avoid leakage.
- Seasonality: use decomposition and seasonal parameters or Prophet components.
- Validation: use walk-forward validation, not random split.

## Deliverables
- Cleaned dataset and data dictionary.
- Notebook/report with EDA, diagnostics, and model comparison.
- Reproducible training scripts.
- Forecast plots and error metrics table.
- Final recommendation with assumptions and risks.

## Project Structure
- `src/preprocess.py`: shared time-series preprocessing.
- `src/run_baseline.py`: naive baseline model.
- `src/run_arima.py`: ARIMA model runner.
- `src/run_prophet.py`: Prophet model runner.
- `src/compare_models.py`: end-to-end model benchmark + reporting artifacts.
- `outputs/`: generated charts, tables, and summary after execution.

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. One-command run (install + compare + open outputs):
   - `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`
3. Run baseline:
   - `python src/run_baseline.py --ticker AAPL --period 5y --test-size 0.2`
4. Run ARIMA:
   - `python src/run_arima.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0`
5. Run Prophet:
   - `python src/run_prophet.py --ticker AAPL --period 5y --test-size 0.2`
6. Compare all models:
   - `python src/compare_models.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0`
   - Artifacts generated:
   - `outputs/forecast_comparison.png`
   - `outputs/model_metrics.csv`
   - `outputs/test_predictions.csv`
   - `outputs/executive_summary.md`

## Demo Talking Points
1. Problem framing: why forecast accuracy matters for planning and risk.
2. Data and preprocessing: daily resampling + missing value handling.
3. Fair evaluation: strict time-based train/test split.
4. Model benchmark: Naive vs ARIMA vs Prophet.
5. Decision output: best model selected by MAE/RMSE and summarized in `executive_summary.md`.
6. Business value: convert forecast quality into operational confidence.

## Portfolio Framing (Resume/Showcase)
Built an end-to-end time-series forecasting pipeline that included preprocessing, stationarity diagnostics, seasonal decomposition, feature engineering, and model benchmarking (ARIMA/Prophet/LSTM) using walk-forward validation to improve forecast reliability for business planning.
