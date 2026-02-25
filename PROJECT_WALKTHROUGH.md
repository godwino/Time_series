# Project Walkthrough (Beginning to End)

## 1) What This Project Does
This project forecasts future stock prices using three models:
- Naive baseline
- ARIMA
- Prophet

It then compares them using MAE and RMSE, ranks performance, and saves professional output files for presentation.

## 2) Project Files and Roles
- `requirements.txt`
  - Python packages needed to run the project.
- `src/preprocess.py`
  - Shared cleaning logic for time series.
  - Handles Yahoo Finance column differences (including MultiIndex format).
- `src/run_baseline.py`
  - Runs the naive model only.
- `src/run_arima.py`
  - Runs ARIMA only.
- `src/run_prophet.py`
  - Runs Prophet only.
- `src/compare_models.py`
  - Runs all models together, ranks metrics, and saves artifacts.
- `outputs/`
  - Generated results (plot, CSVs, and executive summary).
- `PRESENTATION.md`
  - Demo script/talking points.

## 3) Environment Setup
From PowerShell:

```powershell
cd c:\Users\Osayamwen\Desktop\TIME_SERIES
python -m pip install -r requirements.txt
```

## 4) Full End-to-End Run
Run this one command:

```powershell
python src/compare_models.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0
```

What happens in order:
1. Downloads historical prices from Yahoo Finance.
2. Cleans and standardizes the time series (`preprocess.py`).
3. Splits data into train/test by time.
4. Trains and predicts using Naive, ARIMA, and Prophet.
5. Computes MAE and RMSE for each model.
6. Ranks models by error.
7. Saves chart + tables + executive summary.

## 5) Output Files You Get
After running `compare_models.py`, these are created:
- `outputs/forecast_comparison.png`
  - Actual vs predicted lines for all models.
- `outputs/model_metrics.csv`
  - Ranked model errors and `% vs best`.
- `outputs/test_predictions.csv`
  - Actual test values + each model prediction.
- `outputs/executive_summary.md`
  - Business-friendly final summary.

## 6) How to View Everything
List all outputs:

```powershell
Get-ChildItem .\outputs
```

Open plot:

```powershell
start .\outputs\forecast_comparison.png
```

Read summary:

```powershell
Get-Content .\outputs\executive_summary.md
```

Preview metrics:

```powershell
Get-Content .\outputs\model_metrics.csv
```

Preview predictions:

```powershell
Get-Content .\outputs\test_predictions.csv -TotalCount 20
```

## 7) See All Source Code (With Existing Comments)
Show each script in terminal:

```powershell
Get-Content .\src\preprocess.py
Get-Content .\src\run_baseline.py
Get-Content .\src\run_arima.py
Get-Content .\src\run_prophet.py
Get-Content .\src\compare_models.py
```

## 8) Model-by-Model Commands
Naive only:

```powershell
python src/run_baseline.py --ticker AAPL --period 5y --test-size 0.2
```

ARIMA only:

```powershell
python src/run_arima.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0
```

Prophet only:

```powershell
python src/run_prophet.py --ticker AAPL --period 5y --test-size 0.2
```

## 9) If Something Fails
- Install dependencies again:
  - `python -m pip install -r requirements.txt`
- Confirm you are in the correct folder:
  - `cd c:\Users\Osayamwen\Desktop\TIME_SERIES`
- Re-run compare command.

## 10) Presentation Flow (Simple)
1. State business problem.
2. Show methodology (data -> preprocessing -> models -> evaluation).
3. Run `compare_models.py`.
4. Open `forecast_comparison.png`.
5. Open `model_metrics.csv`.
6. Close with `executive_summary.md`.
