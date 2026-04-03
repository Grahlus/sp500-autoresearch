import unittest

import numpy as np
import pandas as pd

from strategies.superstock_entries import build_superstock_breakout_entries
from strategies.superstock_exits import build_superstock_exit_signals, build_superstock_position_mask
from strategies.superstock_weekly import build_superstock_weekly_features


class SuperstockExitTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2023-01-02", periods=320)
        idx = self.dates

        aaa = pd.Series(np.linspace(8.0, 14.0, len(idx)), index=idx)
        aaa.iloc[-5] = 13.8
        aaa.iloc[-4] = 13.7
        aaa.iloc[-3] = 13.4
        aaa.iloc[-2] = 13.0
        aaa.iloc[-1] = 12.6

        bbb = pd.Series(np.linspace(8.0, 12.0, len(idx)), index=idx)
        bbb.iloc[-3:] = [15.5, 15.8, 16.0]

        ccc = pd.Series(np.linspace(8.0, 12.0, len(idx)), index=idx)
        ccc.iloc[-3:] = [11.0, 10.0, 9.0]

        spy = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)

        close = pd.DataFrame({"AAA": aaa, "BBB": bbb, "CCC": ccc, "SPY": spy}, index=idx)
        self.close = close
        self.open_ = close * 0.995
        self.high = close * 1.01
        self.low = close * 0.99
        self.low.iloc[-1, self.low.columns.get_loc("CCC")] = 8.7

        volume = pd.DataFrame(
            {
                "AAA": [120_000] * len(idx),
                "BBB": [120_000] * len(idx),
                "CCC": [120_000] * len(idx),
                "SPY": [1_000_000] * len(idx),
            },
            index=idx,
        )
        volume.iloc[-5:, volume.columns.get_loc("CCC")] = [300_000, 320_000, 340_000, 360_000, 400_000]
        self.volume = volume

        self.vix = pd.Series([20.0] * len(idx), index=idx, name="vix")
        self.data = {
            "open": self.open_,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vix": self.vix,
        }

    def _entries_with_overrides(self):
        weekly = build_superstock_weekly_features(self.data)
        entries = build_superstock_breakout_entries(self.data, weekly=weekly)
        entries.eligible.loc[:, :] = False
        entries.eligible.loc[self.dates[-10], ["AAA", "BBB", "CCC"]] = True
        entries.diagnostics["entry_event"].loc[:, :] = False
        entries.diagnostics["entry_event"].loc[self.dates[-10], ["AAA", "BBB", "CCC"]] = True
        return weekly, entries

    def test_support_break_exit_fires_on_weekly_close_break(self):
        weekly, entries = self._entries_with_overrides()
        exits = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)
        day = self.dates[-1]

        self.assertTrue(bool(exits.rule_masks["support_break_exit"].loc[day, "AAA"]))
        self.assertTrue(bool(exits.eligible.loc[day, "AAA"]))

    def test_parabolic_extension_exit_fires(self):
        weekly, entries = self._entries_with_overrides()
        entries.diagnostics["breakout_pct_above_pivot"].loc[:, "BBB"] = 0.30
        exits = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)
        day = self.dates[-1]

        self.assertTrue(bool(exits.rule_masks["parabolic_extension_exit"].loc[day, "BBB"]))
        self.assertTrue(bool(exits.eligible.loc[day, "BBB"]))

    def test_late_stage_high_volatility_exit_requires_all_conditions(self):
        weekly, entries = self._entries_with_overrides()
        exits = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)
        day = self.dates[-1]

        self.assertTrue(bool(exits.rule_masks["weekly_range_expansion_vs_base"].loc[day, "CCC"]))
        self.assertTrue(bool(exits.rule_masks["weekly_volume_surge_vs_base"].loc[day, "CCC"]))
        self.assertTrue(bool(exits.rule_masks["close_breaks_below_20d_low_excl_today"].loc[day, "CCC"]))
        self.assertTrue(bool(exits.rule_masks["late_stage_hv_exit"].loc[day, "CCC"]))
        self.assertTrue(bool(exits.eligible.loc[day, "CCC"]))

    def test_same_day_entry_suppresses_exit(self):
        weekly, entries = self._entries_with_overrides()
        day = self.dates[-1]
        entries.diagnostics["entry_event"].loc[day, "BBB"] = True
        entries.eligible.loc[day, "BBB"] = True
        entries.diagnostics["breakout_pct_above_pivot"].loc[:, "BBB"] = 0.30
        exits = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)

        self.assertFalse(bool(exits.rule_masks["not_same_day_as_entry"].loc[day, "BBB"]))
        self.assertFalse(bool(exits.eligible.loc[day, "BBB"]))

    def test_future_bar_changes_do_not_change_prior_exit(self):
        weekly, entries = self._entries_with_overrides()
        original = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)

        mutated = {
            "open": self.open_.copy(),
            "high": self.high.copy(),
            "low": self.low.copy(),
            "close": self.close.copy(),
            "volume": self.volume.copy(),
            "vix": self.vix.copy(),
        }
        mutated["close"].iloc[-1, mutated["close"].columns.get_loc("AAA")] = 20.0
        mutated["high"].iloc[-1, mutated["high"].columns.get_loc("AAA")] = 25.0

        weekly_mut = build_superstock_weekly_features(mutated)
        recomputed = build_superstock_exit_signals(mutated, weekly=weekly_mut, entries=entries)
        prior_day = self.dates[-2]

        self.assertEqual(bool(original.eligible.loc[prior_day, "AAA"]), bool(recomputed.eligible.loc[prior_day, "AAA"]))

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
        entries = build_superstock_breakout_entries(data, weekly=weekly)
        exits = build_superstock_exit_signals(data, weekly=weekly, entries=entries)

        self.assertEqual(exits.diagnostics["benchmark_source"], "equal_weight_proxy")

    def test_position_mask_reflects_entry_then_exit(self):
        weekly, entries = self._entries_with_overrides()
        exits = build_superstock_exit_signals(self.data, weekly=weekly, entries=entries)
        positions = build_superstock_position_mask(entries, exits)

        self.assertFalse(bool(positions.loc[self.dates[-10], "AAA"]))
        self.assertTrue(bool(positions.loc[self.dates[-9], "AAA"]))
        if bool(exits.eligible.loc[self.dates[-1], "AAA"]):
            self.assertTrue(bool(positions.loc[self.dates[-1], "AAA"]))


if __name__ == "__main__":
    unittest.main()
