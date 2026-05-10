# Forecast Methodology

This project uses a transparent statistical forecast designed for dashboard reporting rather than opaque black-box prediction.

## Inputs

The model requires a clean date column and one numeric close or price column. Open, high, low, and volume columns are optional and are used for visualization when available.

## Model

Prices are transformed to log scale so trend and volatility are measured in approximate return space. The forecast combines two signals:

- A recency-weighted linear trend over the selected training window
- A short-term median-return drift to dampen unstable trend extrapolation

Daily drift is clipped relative to realized volatility so the model does not project extreme slopes from short noisy windows.

## Uncertainty Bands

Forecast intervals are derived from realized daily volatility and weighted-trend residuals. Bands widen with the square root of the forecast horizon, which is a common approximation for compounding market uncertainty over time.

## Backtest

The dashboard runs a recent walk-forward one-step backtest and reports:

- MAPE: average absolute percent error
- RMSE: root mean squared price error
- Directional accuracy: how often predicted and actual daily direction matched

## Limitations

The model does not include macroeconomic variables, fundamentals, order-book data, news sentiment, regime switching, or corporate actions beyond what is already reflected in the input price series. It should be used as research support, not as a standalone trading system.

