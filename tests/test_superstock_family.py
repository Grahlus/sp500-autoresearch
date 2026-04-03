import unittest

import numpy as np
import pandas as pd

from strategies import get_strategy_family, list_strategy_families
from strategies.superstock import generate_signals


class SuperstockFamilyTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=320)
        idx = self.dates

        aaa = pd.Series(np.linspace(6.0, 12.0, len(idx)), index=idx)
        aaa.iloc[-40:] = np.linspace(11.2, 12.0, 40)
        aaa.iloc[-3] = 9.8
        aaa.iloc[-2] = 10.4
        aaa.iloc[-1] = 10.6

        bbb = pd.Series(np.linspace(20.0, 16.0, len(idx)), index=idx)
        ccc = pd.Series(np.linspace(8.0, 12.0, len(idx)), index=idx)
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
                    "BBB": [25_000] * len(idx),
                    "CCC": [120_000] * len(idx),
                    "SPY": [1_000_000] * len(idx),
                },
                index=idx,
            ),
            "vix": pd.Series([20.0] * len(idx), index=idx, name="vix"),
        }

    def test_strategy_registry_exposes_superstock(self):
        self.assertIn("superstock", list_strategy_families())
        strategy = get_strategy_family("superstock")
        self.assertEqual(strategy.name, "superstock")

    def test_generate_signals_matches_engine_contract(self):
        weights = generate_signals(self.data)

        self.assertEqual(weights.index.tolist(), self.data["close"].index.tolist())
        self.assertEqual(weights.columns.tolist(), self.data["close"].columns.tolist())
        self.assertTrue((weights.fillna(0.0) >= 0.0).all().all())
        self.assertTrue((weights.fillna(0.0) <= 1.0).all().all())

    def test_lifecycle_is_next_day_not_same_day(self):
        weights = generate_signals(self.data)
        self.assertEqual(float(weights.iloc[0].sum()), 0.0)
        self.assertEqual(float(weights.iloc[1].sum()), 0.0)
        self.assertTrue((weights.sum(axis=1) <= 1.0 + 1e-9).all())

    def test_weight_cap_is_respected(self):
        weights = generate_signals(self.data)
        active_counts = (weights > 0).sum(axis=1)
        self.assertTrue((active_counts <= 5).all())


if __name__ == "__main__":
    unittest.main()
