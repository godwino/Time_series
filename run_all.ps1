$ErrorActionPreference = "Stop"

Write-Host "Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "Running full model comparison..." -ForegroundColor Cyan
python .\src\compare_models.py --ticker AAPL --period 5y --test-size 0.2 --p 5 --d 1 --q 0

Write-Host "Opening generated outputs..." -ForegroundColor Cyan
if (Test-Path .\outputs\forecast_comparison.png) {
    Start-Process .\outputs\forecast_comparison.png
}
if (Test-Path .\outputs\executive_summary.md) {
    Start-Process .\outputs\executive_summary.md
}

Write-Host "Done. Check .\outputs for all files." -ForegroundColor Green
