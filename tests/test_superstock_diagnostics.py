import unittest

import numpy as np
import pandas as pd

from strategies.superstock import build_superstock_pipeline, generate_signals
from strategies.superstock_diagnostics import build_superstock_diagnostics


class SuperstockDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=320)
        idx = self.dates

        aaa = pd.Series(np.linspace(6.0, 12.0, len(idx)), index=idx)
        aaa.iloc[-40:] = np.linspace(11.2, 12.0, 40)
        aaa.iloc[-3] = 9.8
        aaa.iloc[-2] = 10.4
        aaa.iloc[-1] = 10.6

        bbb = pd.Series(np.linspace(8.0, 11.0, len(idx)), index=idx)
        bbb.iloc[-40:] = np.linspace(10.2, 10.9, 40)
        bbb.iloc[-3] = 9.7
        bbb.iloc[-2] = 10.3
        bbb.iloc[-1] = 10.4

        ccc = pd.Series(np.linspace(20.0, 16.0, len(idx)), index=idx)
        spy = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)

        close = pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc, "SPY": spy}, index=idx)
        self.data = {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": pd.DataFrame(
                {
                    "AAA": ([250_000] * (len(idx) - 30)) + ([95_000] * 30),
                    "BBB": ([220_000] * (len(idx) - 30)) + ([92_000] * 30),
                    "CCC": [25_000] * len(idx),
                    "SPY": [1_000_000] * len(idx),
                },
                index=idx,
            ),
            "vix": pd.Series([20.0] * len(idx), index=idx, name="vix"),
        }

    def test_pipeline_weights_match_family_signals(self):
        pipeline = build_superstock_pipeline(self.data)
        weights = generate_signals(self.data)

        pd.testing.assert_frame_equal(pipeline["weights"], weights)

    def test_diagnostics_emit_expected_stage_columns(self):
        diag = build_superstock_diagnostics(self.data)
        daily = diag["daily_diagnostics"]

        self.assertIn("screen_pass_count", daily.columns)
        self.assertIn("entry_trigger_count", daily.columns)
        self.assertIn("held_count", daily.columns)
        self.assertIn("exit_count", daily.columns)
        self.assertIn("gross_exposure", daily.columns)
        self.assertIn("simultaneous_positions", daily.columns)
        self.assertTrue((daily["entry_trigger_count"] <= daily["screen_pass_count"]).all())
        self.assertTrue((daily["held_count"] >= 0).all())

    def test_missing_vix_uses_unknown_regime_and_fallback_benchmark_stays_explicit(self):
        data = {
            "open": self.data["open"].drop(columns=["SPY"]),
            "high": self.data["high"].drop(columns=["SPY"]),
            "low": self.data["low"].drop(columns=["SPY"]),
            "close": self.data["close"].drop(columns=["SPY"]),
            "volume": self.data["volume"].drop(columns=["SPY"]),
        }
        diag = build_superstock_diagnostics(data)

        self.assertEqual(diag["metadata"]["benchmark_source"], "equal_weight_proxy")
        self.assertTrue((diag["daily_diagnostics"]["vix_regime"] == "unknown").all())
        self.assertIn("benchmark_source", diag["summary"]["metric"].values)

    def test_start_date_filters_daily_and_trade_outputs(self):
        start_date = self.dates[-40]
        diag = build_superstock_diagnostics(self.data, start_date=start_date)

        self.assertTrue((pd.to_datetime(diag["daily_diagnostics"]["date"]) >= start_date).all())
        if not diag["trade_attribution"].empty:
            entry_exec = pd.to_datetime(diag["trade_attribution"]["entry_exec_date"], errors="coerce")
            self.assertTrue((entry_exec.dropna() >= start_date).all())


if __name__ == "__main__":
    unittest.main()
