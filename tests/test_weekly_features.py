import unittest

import pandas as pd

from strategies.superstock_weekly import build_superstock_weekly_features, to_daily_feature_map
from strategies.weekly import align_completed_weekly_to_daily, build_weekly_bars


class WeeklyFeatureLayerTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=15)
        idx = self.dates
        cols = ["AAA", "SPY"]

        self.open_ = pd.DataFrame(
            {
                "AAA": range(10, 25),
                "SPY": range(100, 115),
            },
            index=idx,
        )
        self.high = self.open_ + 1
        self.low = self.open_ - 1
        self.close = self.open_ + 0.5
        self.volume = pd.DataFrame(
            {
                "AAA": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
                "SPY": [1000] * 15,
            },
            index=idx,
        )
        self.vix = pd.Series(range(20, 35), index=idx, name="vix")

    def test_build_weekly_bars_aggregates_trading_weeks(self):
        weekly = build_weekly_bars(self.open_, self.high, self.low, self.close, self.volume)

        self.assertEqual(len(weekly.close), 3)
        self.assertEqual(weekly.close.index[0], pd.Timestamp("2024-01-05"))
        self.assertEqual(weekly.close.index[1], pd.Timestamp("2024-01-12"))
        self.assertEqual(weekly.close.index[2], pd.Timestamp("2024-01-19"))

        self.assertEqual(weekly.open.loc[pd.Timestamp("2024-01-05"), "AAA"], 10)
        self.assertEqual(weekly.high.loc[pd.Timestamp("2024-01-05"), "AAA"], 15)
        self.assertEqual(weekly.low.loc[pd.Timestamp("2024-01-05"), "AAA"], 9)
        self.assertEqual(weekly.close.loc[pd.Timestamp("2024-01-05"), "AAA"], 14.5)
        self.assertEqual(weekly.volume.loc[pd.Timestamp("2024-01-05"), "AAA"], 600)

        self.assertTrue(weekly.week_end_mask.loc[pd.Timestamp("2024-01-05")])
        self.assertFalse(weekly.week_end_mask.loc[pd.Timestamp("2024-01-04")])

    def test_align_completed_weekly_to_daily_uses_last_completed_week(self):
        weekly = build_weekly_bars(self.open_, self.high, self.low, self.close, self.volume)
        weekly_feature = weekly.close[["AAA"]]
        aligned = align_completed_weekly_to_daily(weekly_feature, self.dates, weekly.close.index)

        self.assertTrue(pd.isna(aligned.loc[pd.Timestamp("2024-01-04"), "AAA"]))
        self.assertEqual(aligned.loc[pd.Timestamp("2024-01-05"), "AAA"], 14.5)
        self.assertEqual(aligned.loc[pd.Timestamp("2024-01-08"), "AAA"], 14.5)
        self.assertEqual(aligned.loc[pd.Timestamp("2024-01-12"), "AAA"], 19.5)

    def test_build_weekly_bars_handles_short_holiday_week(self):
        idx = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05"])
        open_ = pd.DataFrame({"AAA": [10, 11, 12, 13]}, index=idx)
        high = open_ + 1
        low = open_ - 1
        close = open_ + 0.5
        volume = pd.DataFrame({"AAA": [100, 200, 300, 400]}, index=idx)

        weekly = build_weekly_bars(open_, high, low, close, volume)

        self.assertEqual(len(weekly.close), 1)
        self.assertEqual(weekly.close.index[0], pd.Timestamp("2024-07-05"))
        self.assertEqual(weekly.volume.iloc[0]["AAA"], 1000)

    def test_superstock_feature_bundle_and_daily_alignment(self):
        data = {
            "open": self.open_,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vix": self.vix,
        }
        features = build_superstock_weekly_features(data)
        aligned = to_daily_feature_map(features, self.dates)

        self.assertIn("AAA", features.weekly_close.columns)
        self.assertIn("SPY", features.weekly_close.columns)
        self.assertIn("AAA", features.rs_ratio_spy.columns)
        self.assertEqual(features.spy_weekly_close.index.tolist(), features.weekly_close.index.tolist())
        self.assertEqual(features.vix_weekly_close.index.tolist(), features.weekly_close.index.tolist())
        self.assertEqual(aligned["weekly_close"].loc[pd.Timestamp("2024-01-08"), "AAA"], 14.5)
        self.assertTrue(aligned["week_end_mask"].loc[pd.Timestamp("2024-01-05")])


if __name__ == "__main__":
    unittest.main()
