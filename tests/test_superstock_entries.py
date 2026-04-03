import unittest

import numpy as np
import pandas as pd

from strategies.screening import ScreeningResult
from strategies.superstock_entries import build_superstock_breakout_entries, build_superstock_entry_weights
from strategies.superstock_weekly import build_superstock_weekly_features


class SuperstockEntryTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=320)
        idx = self.dates

        aaa = pd.Series(9.6, index=idx)
        aaa.iloc[-30:-2] = np.linspace(9.55, 9.9, 28)
        aaa.iloc[-3] = 9.8
        aaa.iloc[-2] = 10.4
        aaa.iloc[-1] = 10.6

        bbb = pd.Series(9.7, index=idx)
        bbb.iloc[-30:-2] = np.linspace(9.65, 9.95, 28)
        bbb.iloc[-3] = 9.82
        bbb.iloc[-2] = 10.4
        bbb.iloc[-1] = 10.5

        spy = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)

        close = pd.DataFrame({"AAA": aaa, "BBB": bbb, "SPY": spy}, index=idx)
        self.close = close
        self.open_ = close * 0.995
        self.high = close * 1.01
        self.low = close * 0.99

        aaa_volume = [100_000] * len(idx)
        bbb_volume = [100_000] * len(idx)
        aaa_volume[-2] = 250_000
        aaa_volume[-1] = 260_000
        bbb_volume[-2] = 120_000
        bbb_volume[-1] = 125_000

        self.volume = pd.DataFrame(
            {
                "AAA": aaa_volume,
                "BBB": bbb_volume,
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

    def _screen_result(self, eligible_cols: dict[str, bool]) -> ScreeningResult:
        mask = pd.DataFrame(False, index=self.dates, columns=self.close.columns)
        for symbol, allowed in eligible_cols.items():
            mask[symbol] = allowed
        rs_26 = pd.DataFrame(0.0, index=self.dates, columns=self.close.columns)
        rs_52 = pd.DataFrame(0.0, index=self.dates, columns=self.close.columns)
        rs_26["AAA"] = 0.95
        rs_26["BBB"] = 0.80
        rs_52["AAA"] = 0.90
        rs_52["BBB"] = 0.75
        return ScreeningResult(
            eligible=mask,
            rule_masks={"screen_eligible": mask},
            diagnostics={
                "benchmark_source": "spy_symbol",
                "benchmark_weekly_close": None,
                "spy_weekly_close": None,
                "rs_rank_26w": rs_26,
                "rs_rank_52w": rs_52,
            },
        )

    def test_breakout_event_fires_once_on_crossing_day(self):
        weekly = build_superstock_weekly_features(self.data)
        screen = self._screen_result({"AAA": True, "BBB": True, "SPY": False})
        entries = build_superstock_breakout_entries(self.data, weekly=weekly, screen=screen)

        signal_day = self.dates[-2]
        next_day = self.dates[-1]
        self.assertTrue(bool(entries.eligible.loc[signal_day, "AAA"]))
        self.assertFalse(bool(entries.eligible.loc[next_day, "AAA"]))

    def test_volume_expansion_blocks_candidate_without_real_expansion(self):
        weekly = build_superstock_weekly_features(self.data)
        screen = self._screen_result({"AAA": True, "BBB": True, "SPY": False})
        entries = build_superstock_breakout_entries(self.data, weekly=weekly, screen=screen)
        signal_day = self.dates[-2]

        self.assertTrue(bool(entries.rule_masks["daily_volume_expansion"].loc[signal_day, "AAA"]))
        self.assertFalse(bool(entries.rule_masks["daily_volume_expansion"].loc[signal_day, "BBB"]))
        self.assertFalse(bool(entries.eligible.loc[signal_day, "BBB"]))

    def test_screen_gate_blocks_entries(self):
        weekly = build_superstock_weekly_features(self.data)
        screen = self._screen_result({"AAA": False, "BBB": True, "SPY": False})
        entries = build_superstock_breakout_entries(self.data, weekly=weekly, screen=screen)
        signal_day = self.dates[-2]

        self.assertFalse(bool(entries.eligible.loc[signal_day, "AAA"]))

    def test_future_bar_changes_do_not_change_prior_entry(self):
        weekly = build_superstock_weekly_features(self.data)
        screen = self._screen_result({"AAA": True, "BBB": True, "SPY": False})
        original = build_superstock_breakout_entries(self.data, weekly=weekly, screen=screen)

        mutated = {
            "open": self.open_.copy(),
            "high": self.high.copy(),
            "low": self.low.copy(),
            "close": self.close.copy(),
            "volume": self.volume.copy(),
            "vix": self.vix.copy(),
        }
        mutated["close"].iloc[-1, mutated["close"].columns.get_loc("AAA")] = 25.0
        mutated["high"].iloc[-1, mutated["high"].columns.get_loc("AAA")] = 30.0
        mutated["volume"].iloc[-1, mutated["volume"].columns.get_loc("AAA")] = 5_000_000

        weekly_mut = build_superstock_weekly_features(mutated)
        recomputed = build_superstock_breakout_entries(mutated, weekly=weekly_mut, screen=screen)
        signal_day = self.dates[-2]

        self.assertEqual(bool(original.eligible.loc[signal_day, "AAA"]), bool(recomputed.eligible.loc[signal_day, "AAA"]))

    def test_benchmark_source_remains_explicit_under_fallback(self):
        data = {
            "open": self.open_.drop(columns=["SPY"]),
            "high": self.high.drop(columns=["SPY"]),
            "low": self.low.drop(columns=["SPY"]),
            "close": self.close.drop(columns=["SPY"]),
            "volume": self.volume.drop(columns=["SPY"]),
            "vix": self.vix,
        }
        weekly = build_superstock_weekly_features(data)
        mask = pd.DataFrame(True, index=self.dates, columns=data["close"].columns)
        rs = pd.DataFrame(0.8, index=self.dates, columns=data["close"].columns)
        screen = ScreeningResult(
            eligible=mask,
            rule_masks={"screen_eligible": mask},
            diagnostics={
                "benchmark_source": weekly.benchmark_source,
                "benchmark_weekly_close": None,
                "spy_weekly_close": None,
                "rs_rank_26w": rs,
                "rs_rank_52w": rs,
            },
        )
        entries = build_superstock_breakout_entries(data, weekly=weekly, screen=screen)

        self.assertEqual(entries.diagnostics["benchmark_source"], "equal_weight_proxy")

    def test_entry_weights_rank_and_cap_positions(self):
        weekly = build_superstock_weekly_features(self.data)
        screen = self._screen_result({"AAA": True, "BBB": True, "SPY": False})
        entries = build_superstock_breakout_entries(self.data, weekly=weekly, screen=screen)
        weights = build_superstock_entry_weights(self.data, entries=entries, max_positions=1)
        signal_day = self.dates[-2]

        self.assertEqual(float(weights.loc[signal_day, "AAA"]), 1.0)
        self.assertEqual(float(weights.loc[signal_day, "BBB"]), 0.0)


if __name__ == "__main__":
    unittest.main()
