from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from stock_dashboard.forecast import build_forecast
from stock_dashboard.data import ColumnMap, standardize_prices
from stock_dashboard.metrics import compute_kpis, max_drawdown


class ForecastMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=220)
        trend = np.linspace(100, 132, len(dates))
        seasonal = np.sin(np.arange(len(dates)) / 7) * 1.5
        close = trend + seasonal
        self.price_df = pd.DataFrame({"Date": dates, "Close": close, "Volume": 1_000_000})

    def test_build_forecast_returns_expected_horizon(self) -> None:
        result = build_forecast(self.price_df, horizon=20, training_window=120)

        self.assertEqual(len(result.forecast), 20)
        self.assertTrue((result.forecast["Lower_95"] <= result.forecast["Lower_80"]).all())
        self.assertTrue((result.forecast["Lower_80"] <= result.forecast["Forecast"]).all())
        self.assertTrue((result.forecast["Forecast"] <= result.forecast["Upper_80"]).all())
        self.assertTrue((result.forecast["Upper_80"] <= result.forecast["Upper_95"]).all())
        self.assertIn("backtest_mape", result.summary)

    def test_compute_kpis_has_core_fields(self) -> None:
        kpis = compute_kpis(self.price_df)

        self.assertGreater(kpis["last_close"], 0)
        self.assertGreater(kpis["observations"], 200)
        self.assertIn("volatility", kpis)
        self.assertIn("max_drawdown", kpis)

    def test_max_drawdown_detects_loss_from_peak(self) -> None:
        close = pd.Series([100, 120, 90, 110])

        self.assertAlmostEqual(max_drawdown(close), -0.25)

    def test_standardize_prices_parses_yyyymmdd_integers(self) -> None:
        raw = pd.DataFrame(
            {
                "ticker": ["ABC", "ABC"],
                "date": [20260102, 20260105],
                "close": [10.0, 11.0],
            }
        )

        clean = standardize_prices(raw, ColumnMap(date="date", ticker="ticker", close="close"), ticker="ABC")

        self.assertEqual(clean["Date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-02", "2026-01-05"])


if __name__ == "__main__":
    unittest.main()
