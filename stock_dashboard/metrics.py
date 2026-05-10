from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def pct_change(current: float, previous: float) -> float:
    if previous == 0 or math.isnan(previous):
        return float("nan")
    return (current / previous) - 1.0


def max_drawdown(close: pd.Series) -> float:
    prices = close.dropna().astype(float)
    if prices.empty:
        return float("nan")
    cumulative_peak = prices.cummax()
    drawdown = prices / cumulative_peak - 1.0
    return float(drawdown.min())


def drawdown_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    work = price_df[["Date", "Close"]].dropna().copy()
    work["Peak"] = work["Close"].cummax()
    work["Drawdown"] = work["Close"] / work["Peak"] - 1.0
    return work[["Date", "Drawdown"]]


def period_return(price_df: pd.DataFrame, periods: int) -> float:
    close = price_df["Close"].dropna()
    if len(close) <= periods:
        return float("nan")
    return pct_change(float(close.iloc[-1]), float(close.iloc[-periods - 1]))


def year_to_date_return(price_df: pd.DataFrame) -> float:
    work = price_df[["Date", "Close"]].dropna()
    if work.empty:
        return float("nan")
    last_year = work["Date"].iloc[-1].year
    ytd = work[work["Date"].dt.year == last_year]
    if len(ytd) < 2:
        return float("nan")
    return pct_change(float(ytd["Close"].iloc[-1]), float(ytd["Close"].iloc[0]))


def compute_kpis(price_df: pd.DataFrame) -> dict[str, Any]:
    close = price_df["Close"].dropna().astype(float)
    returns = close.pct_change().dropna()

    last_close = float(close.iloc[-1]) if len(close) else float("nan")
    previous_close = float(close.iloc[-2]) if len(close) > 1 else float("nan")
    daily_return = pct_change(last_close, previous_close)

    volatility = float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) > 1 else float("nan")
    sharpe = (
        float((returns.mean() / returns.std()) * np.sqrt(TRADING_DAYS))
        if len(returns) > 1 and returns.std() != 0
        else float("nan")
    )

    if len(close) > 1:
        date_span = max((price_df["Date"].iloc[-1] - price_df["Date"].iloc[0]).days, 1)
        years = date_span / 365.25
        cagr = (last_close / float(close.iloc[0])) ** (1 / years) - 1 if years > 0 and close.iloc[0] > 0 else float("nan")
    else:
        cagr = float("nan")

    high_52w = float(close.tail(TRADING_DAYS).max()) if len(close) else float("nan")
    low_52w = float(close.tail(TRADING_DAYS).min()) if len(close) else float("nan")
    avg_volume = (
        float(price_df["Volume"].tail(30).mean())
        if "Volume" in price_df.columns and price_df["Volume"].notna().any()
        else float("nan")
    )

    return {
        "last_close": last_close,
        "daily_return": daily_return,
        "return_1m": period_return(price_df, 21),
        "return_3m": period_return(price_df, 63),
        "return_6m": period_return(price_df, 126),
        "return_ytd": year_to_date_return(price_df),
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(close),
        "cagr": float(cagr),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "avg_volume_30d": avg_volume,
        "observations": int(len(price_df)),
        "start_date": price_df["Date"].iloc[0] if len(price_df) else None,
        "end_date": price_df["Date"].iloc[-1] if len(price_df) else None,
    }


def format_money(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    return f"{value:.2%}"


def format_number(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "n/a"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"

