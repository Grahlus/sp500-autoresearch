import unittest

import pandas as pd

from prepare import load_data
from strategies import get_strategy_family, list_strategy_families


def _slice_data(data: dict, days: int = 400, tickers: int = 60) -> dict:
    close = data["close"]
    idx = close.index[:days]
    cols = list(close.columns[:tickers])
    if "SPY" in close.columns and "SPY" not in cols:
        cols = ["SPY"] + [col for col in cols if col != "SPY"]
    sliced: dict = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            common_cols = [col for col in cols if col in value.columns]
            sliced[key] = value.loc[idx, common_cols] if common_cols else value.loc[idx]
        elif isinstance(value, pd.Series):
            sliced[key] = value.loc[idx]
        else:
            sliced[key] = value
    sliced["index"] = idx
    sliced["train_end"] = idx[-1]
    return sliced


class ResearchFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _slice_data(load_data())

    def test_registry_includes_ml_and_rl_families(self):
        families = list_strategy_families()
        self.assertIn("ml_ranker", families)
        self.assertIn("rl_bandit", families)

    def test_ml_ranker_generates_weights(self):
        family = get_strategy_family("ml_ranker")
        self.assertEqual(family.name, "ml_ranker")
        weights = family.generate_signals_with_config(
            self.data,
            {
                "model_type": "ridge",
                "lookback_days": 252,
                "horizon_days": 10,
                "rebalance_days": 5,
                "top_pct": 0.03,
                "max_positions": 10,
                "feature_set": "trend_volume",
                "allow_short": False,
                "use_vix_gate": True,
                "use_fear_greed_gate": False,
            },
        )
        self.assertEqual(weights.shape, self.data["close"].shape)
        self.assertTrue((weights.fillna(0.0) >= 0).all().all())

    def test_rl_bandit_generates_weights(self):
        family = get_strategy_family("rl_bandit")
        self.assertEqual(family.name, "rl_bandit")
        weights = family.generate_signals_with_config(
            self.data,
            {
                "policy_type": "ucb",
                "lookback_days": 252,
                "rebalance_days": 5,
                "epsilon": 0.10,
                "ucb_bonus": 1.0,
                "max_positions": 5,
                "momentum_top_pct": 0.03,
                "superstock_top_pct": 0.03,
                "use_vix_gate": True,
                "use_fear_greed_gate": True,
            },
        )
        self.assertEqual(weights.shape, self.data["close"].shape)
        self.assertTrue((weights.fillna(0.0) >= 0).all().all())


if __name__ == "__main__":
    unittest.main()
