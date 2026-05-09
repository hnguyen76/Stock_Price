# Stock Forecast Intelligence

Professional Streamlit dashboard for stock price analysis, scenario forecasting, risk review, and downloadable research notes.

## Highlights

- Interactive ticker, date range, and forecast horizon controls
- Automatic column detection for common OHLCV stock datasets
- Candlestick or close-price chart with MA20 and MA60 overlays
- Base, bear, and bull forecast scenarios with 80% and 95% bands
- Risk layer: volatility, drawdown, return distribution, Sharpe proxy, and backtest quality
- Markdown report export for sharing forecast assumptions and headline metrics

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

On Streamlit Cloud, the app opens with a bundled sample dataset covering AAPL, AMZN, GOOGL, META, MSFT, NVDA, and TSLA. When running locally, the app looks for `full_data.parquet` in the repository root by default. You can also load a local path or upload a parquet, CSV, TSV, or Excel file from the sidebar.

Local datasets are ignored by default so source code can be published without uploading private or oversized market data. Although the Streamlit uploader is configured for files up to 1000MB, uploading a 627MB parquet file can still exceed Community Cloud memory during parsing. For the full dataset, run locally or publish the data through a dedicated storage path.

## Data Expectations

The dashboard works best with daily OHLCV data. Column names are auto-detected when they resemble:

- Date: `date`, `datetime`, `timestamp`, `trade_date`
- Ticker: `ticker`, `symbol`, `stock`, `code`
- Prices: `open`, `high`, `low`, `close`, `adj_close`, `price`
- Volume: `volume`, `vol`, `shares`

If your dataset uses different names, use the sidebar column mapping controls.

## Forecast Method

The forecast model is intentionally transparent:

1. Convert prices to log scale.
2. Fit a recency-weighted trend over the selected training window.
3. Blend trend with recent median-return drift to reduce overreaction.
4. Build uncertainty bands from realized volatility and trend residuals.
5. Backtest recent one-step forecasts to estimate MAPE, RMSE, and directional accuracy.

This is decision support for analysis and reporting, not investment advice.

## Git LFS

Large parquet datasets are ignored by default. If you intentionally want to publish a dataset, track it with Git LFS and force-add the file:

```powershell
git lfs install
git lfs track "*.parquet"
git add -f full_data.parquet
```

The repository is configured to support this workflow.
