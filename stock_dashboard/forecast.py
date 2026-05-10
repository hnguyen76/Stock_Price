from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    forecast: pd.DataFrame
    backtest: pd.DataFrame
    summary: dict[str, float]


def _safe_log_prices(close: pd.Series) -> np.ndarray:
    prices = close.astype(float).clip(lower=1e-9)
    return np.log(prices.to_numpy())


def _fit_weighted_trend(log_prices: np.ndarray) -> tuple[float, float, float, float]:
    n = len(log_prices)
    if n < 3:
        return float(log_prices[-1]), 0.0, 0.0, 0.0

    x = np.arange(n, dtype=float)
    weights = np.linspace(0.35, 1.0, n)
    slope, intercept = np.polyfit(x, log_prices, deg=1, w=weights)
    fitted = intercept + slope * x
    residuals = log_prices - fitted
    residual_std = float(np.std(residuals, ddof=1)) if n > 2 else 0.0
    total = float(np.sum((log_prices - np.mean(log_prices)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / total if total > 0 else 0.0
    return float(intercept), float(slope), max(residual_std, 0.0), max(min(r2, 1.0), 0.0)


def _predict_log_path(close: pd.Series, horizon: int, training_window: int) -> tuple[np.ndarray, float, float]:
    clean_close = close.dropna().astype(float)
    window = clean_close.tail(max(training_window, 10))
    log_prices = _safe_log_prices(window)
    returns = np.diff(log_prices)

    intercept, slope, residual_std, r2 = _fit_weighted_trend(log_prices)
    daily_vol = float(np.std(returns, ddof=1)) if len(returns) > 2 else residual_std
    daily_vol = max(daily_vol, residual_std, 1e-4)

    max_daily_drift = max(0.0025, daily_vol * 0.35)
    clipped_slope = float(np.clip(slope, -max_daily_drift, max_daily_drift))

    median_return = float(np.median(returns[-21:])) if len(returns) else 0.0
    median_return = float(np.clip(median_return, -max_daily_drift, max_daily_drift))

    steps = np.arange(1, horizon + 1, dtype=float)
    last_log = log_prices[-1]
    trend_path = last_log + clipped_slope * steps
    drift_path = last_log + median_return * np.sqrt(steps)

    trend_weight = 0.55 + 0.35 * r2
    forecast_log = trend_weight * trend_path + (1.0 - trend_weight) * drift_path

    # Keep a lightly anchored version of the fitted line in the blend when the trend is stable.
    fitted_future = intercept + clipped_slope * (len(log_prices) - 1 + steps)
    forecast_log = 0.85 * forecast_log + 0.15 * fitted_future
    return forecast_log, daily_vol, r2


def _future_business_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=horizon)


def build_forecast(price_df: pd.DataFrame, horizon: int = 30, training_window: int = 252) -> ForecastResult:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(price_df) < 12:
        raise ValueError("At least 12 observations are required for a useful forecast.")

    history = price_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    forecast_log, daily_vol, r2 = _predict_log_path(history["Close"], horizon, training_window)
    steps = np.arange(1, horizon + 1, dtype=float)
    interval_scale = daily_vol * np.sqrt(steps)

    forecast = pd.DataFrame(
        {
            "Date": _future_business_dates(history["Date"].iloc[-1], horizon),
            "Forecast": np.exp(forecast_log),
            "Lower_80": np.exp(forecast_log - 1.2816 * interval_scale),
            "Upper_80": np.exp(forecast_log + 1.2816 * interval_scale),
            "Lower_95": np.exp(forecast_log - 1.96 * interval_scale),
            "Upper_95": np.exp(forecast_log + 1.96 * interval_scale),
            "Bear": np.exp(forecast_log - 0.75 * interval_scale),
            "Bull": np.exp(forecast_log + 0.75 * interval_scale),
        }
    )
    backtest = walk_forward_backtest(history, training_window=training_window)
    summary = forecast_summary(history, forecast, backtest, daily_vol=daily_vol, trend_r2=r2)
    return ForecastResult(forecast=forecast, backtest=backtest, summary=summary)


def walk_forward_backtest(price_df: pd.DataFrame, training_window: int = 252, test_size: int | None = None) -> pd.DataFrame:
    history = price_df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    n = len(history)
    if n < 40:
        return pd.DataFrame(columns=["Date", "Actual", "Predicted", "Error", "Absolute_Percent_Error"])

    if test_size is None:
        test_size = min(90, max(20, n // 5))
    min_train = min(max(30, training_window // 2), max(30, n - test_size))
    start = max(min_train, n - test_size)

    rows = []
    for idx in range(start, n):
        train = history.iloc[:idx]
        actual = float(history["Close"].iloc[idx])
        predicted_log, _, _ = _predict_log_path(train["Close"], 1, training_window)
        predicted = float(np.exp(predicted_log[0]))
        rows.append(
            {
                "Date": history["Date"].iloc[idx],
                "Actual": actual,
                "Predicted": predicted,
                "Error": predicted - actual,
                "Absolute_Percent_Error": abs(predicted / actual - 1.0) if actual else math.nan,
            }
        )

    return pd.DataFrame(rows)


def forecast_summary(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    backtest: pd.DataFrame,
    daily_vol: float,
    trend_r2: float,
) -> dict[str, float]:
    last_close = float(history["Close"].iloc[-1])
    target = float(forecast["Forecast"].iloc[-1])
    expected_return = target / last_close - 1.0 if last_close else math.nan

    if backtest.empty:
        mape = rmse = directional_accuracy = math.nan
    else:
        mape = float(backtest["Absolute_Percent_Error"].mean())
        rmse = float(np.sqrt(np.mean(backtest["Error"] ** 2)))
        actual_direction = np.sign(backtest["Actual"].diff())
        predicted_direction = np.sign(backtest["Predicted"].diff())
        comparable = pd.DataFrame({"actual": actual_direction, "predicted": predicted_direction}).dropna()
        directional_accuracy = (
            float((comparable["actual"] == comparable["predicted"]).mean()) if not comparable.empty else math.nan
        )

    return {
        "last_close": last_close,
        "target_price": target,
        "expected_return": expected_return,
        "daily_volatility": daily_vol,
        "annualized_volatility": daily_vol * np.sqrt(252),
        "trend_r2": trend_r2,
        "backtest_mape": mape,
        "backtest_rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }

