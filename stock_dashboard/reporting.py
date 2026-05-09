from __future__ import annotations

from datetime import datetime
import math
from typing import Any

import pandas as pd

from .metrics import format_money, format_number, format_percent


def _fmt(value: float, kind: str = "percent") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "money":
        return format_money(float(value))
    if kind == "number":
        return format_number(float(value))
    return format_percent(float(value))


def build_markdown_report(
    ticker: str,
    kpis: dict[str, Any],
    forecast_summary: dict[str, float],
    forecast: pd.DataFrame,
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.now()
    final_row = forecast.iloc[-1]
    report_date = pd.Timestamp(final_row["Date"]).date().isoformat()

    return f"""# Stock Forecast Report: {ticker}

Generated: {generated_at:%Y-%m-%d %H:%M}

## Executive Summary

- Latest close: {_fmt(kpis.get("last_close"), "money")}
- Forecast target ({report_date}): {_fmt(forecast_summary.get("target_price"), "money")}
- Expected move: {_fmt(forecast_summary.get("expected_return"))}
- 80% forecast range: {_fmt(float(final_row["Lower_80"]), "money")} to {_fmt(float(final_row["Upper_80"]), "money")}
- Annualized volatility: {_fmt(kpis.get("volatility"))}

## Historical Profile

- Observations: {_fmt(kpis.get("observations"), "number")}
- 1M return: {_fmt(kpis.get("return_1m"))}
- 3M return: {_fmt(kpis.get("return_3m"))}
- 6M return: {_fmt(kpis.get("return_6m"))}
- YTD return: {_fmt(kpis.get("return_ytd"))}
- Max drawdown: {_fmt(kpis.get("max_drawdown"))}
- Sharpe proxy: {kpis.get("sharpe", float("nan")):.2f}

## Forecast Quality

- Backtest MAPE: {_fmt(forecast_summary.get("backtest_mape"))}
- Backtest RMSE: {_fmt(forecast_summary.get("backtest_rmse"), "money")}
- Directional accuracy: {_fmt(forecast_summary.get("directional_accuracy"))}
- Trend fit R-squared: {forecast_summary.get("trend_r2", float("nan")):.2f}

## Method Note

The model blends a recent weighted log-price trend with a short-term median-return drift and derives forecast bands from realized volatility and trend residuals. Treat the forecast as decision support, not investment advice.
"""

