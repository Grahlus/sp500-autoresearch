import unittest

import numpy as np
import pandas as pd

from strategies.superstock_screen import build_base_building_mask, build_liquidity_and_price_mask, build_superstock_screen
from strategies.superstock_trend import build_superstock_trend_template
from strategies.superstock_weekly import build_superstock_weekly_features, to_daily_feature_map


class SuperstockScreeningTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=320)
        idx = self.dates

        aaa = pd.Series(np.linspace(6.0, 12.0, len(idx)), index=idx)
        aaa.iloc[-40:] = np.linspace(11.2, 12.0, 40)
        bbb = pd.Series(np.linspace(20.0, 16.0, len(idx)), index=idx)
        spy = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
        aaa_volume = ([250_000] * (len(idx) - 30)) + ([95_000] * 30)

        close = pd.DataFrame({"AAA": aaa, "BBB": bbb, "SPY": spy}, index=idx)
        self.close = close
        self.open_ = close * 0.995
        self.high = close * 1.01
        self.low = close * 0.99
        self.volume = pd.DataFrame(
            {
                "AAA": aaa_volume,
                "BBB": [25_000] * len(idx),
                "SPY": [1_000_000] * len(idx),
            },
            index=idx,
        )
        self.vix = pd.Series([20.0] * len(idx), index=idx, name="vix")
        self.data = {
            "open": self.open_,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vix": self.vix,
        }

    def test_trend_template_emits_benchmark_explicitly_and_is_daily_aligned(self):
        weekly = build_superstock_weekly_features(self.data)
        result = build_superstock_trend_template(self.data, weekly)
        aligned = to_daily_feature_map(weekly, self.dates)

        monday = pd.Timestamp("2024-03-18")
        friday = pd.Timestamp("2024-03-15")
        self.assertEqual(result.diagnostics["benchmark_source"], "spy_symbol")
        self.assertEqual(
            result.diagnostics["benchmark_weekly_close"].loc[monday],
            aligned["benchmark_weekly_close"].loc[friday],
        )
        self.assertIn("weekly_above_10w", result.rule_masks)

    def test_trend_template_passes_aaa_and_rejects_bbb_late_in_sample(self):
        weekly = build_superstock_weekly_features(self.data)
        result = build_superstock_trend_template(self.data, weekly)
        day = self.dates[-1]

        self.assertTrue(bool(result.rule_masks["daily_above_50d"].loc[day, "AAA"]))
        self.assertTrue(bool(result.rule_masks["ma_stack_bullish"].loc[day, "AAA"]))
        self.assertTrue(bool(result.eligible.loc[day, "AAA"]))
        self.assertFalse(bool(result.rule_masks["daily_above_50d"].loc[day, "BBB"]))
        self.assertFalse(bool(result.eligible.loc[day, "BBB"]))

    def test_liquidity_and_price_screen_uses_band_and_dollar_volume(self):
        result = build_liquidity_and_price_mask(self.data)
        day = self.dates[-1]

        self.assertTrue(bool(result.rule_masks["price_in_superstock_band"].loc[day, "AAA"]))
        self.assertFalse(bool(result.rule_masks["price_in_superstock_band"].loc[day, "BBB"]))
        self.assertTrue(bool(result.rule_masks["min_dollar_volume"].loc[day, "AAA"]))
        self.assertFalse(bool(result.rule_masks["min_dollar_volume"].loc[day, "BBB"]))

    def test_base_building_and_final_screen_work_without_fundamentals(self):
        weekly = build_superstock_weekly_features(self.data)
        base = build_base_building_mask(self.data, weekly)
        result = build_superstock_screen(self.data, weekly=weekly, fundamentals=None)
        day = self.dates[-1]

        self.assertIn("weekly_range_not_exploding", base.rule_masks)
        self.assertIsNone(result.diagnostics["fundamentals_used"])
        self.assertTrue(bool(result.rule_masks["vix_not_extreme"].loc[day, "AAA"]))
        self.assertTrue(bool(result.eligible.loc[day, "AAA"]))
        self.assertFalse(bool(result.eligible.loc[day, "BBB"]))

    def test_missing_vix_does_not_crash_and_is_permissive(self):
        data = {k: v for k, v in self.data.items() if k != "vix"}
        result = build_superstock_screen(data)
        day = self.dates[-1]

        self.assertTrue(bool(result.rule_masks["vix_not_extreme"].loc[day, "AAA"]))

    def test_insufficient_history_remains_present_but_ineligible(self):
        short_dates = self.dates[:120]
        close = self.close.loc[short_dates]
        data = {
            "open": self.open_.loc[short_dates],
            "high": self.high.loc[short_dates],
            "low": self.low.loc[short_dates],
            "close": close,
            "volume": self.volume.loc[short_dates],
            "vix": self.vix.loc[short_dates],
        }
        result = build_superstock_screen(data)
        day = short_dates[-1]

        self.assertIn("AAA", result.eligible.columns)
        self.assertFalse(bool(result.rule_masks["has_min_daily_history"].loc[day, "AAA"]))
        self.assertFalse(bool(result.eligible.loc[day, "AAA"]))

    def test_benchmark_fallback_remains_explicit(self):
        data = {
            "open": self.open_.drop(columns=["SPY"]),
            "high": self.high.drop(columns=["SPY"]),
            "low": self.low.drop(columns=["SPY"]),
            "close": self.close.drop(columns=["SPY"]),
            "volume": self.volume.drop(columns=["SPY"]),
            "vix": self.vix,
        }
        result = build_superstock_screen(data)

        self.assertEqual(result.diagnostics["benchmark_source"], "equal_weight_proxy")


if __name__ == "__main__":
    unittest.main()
