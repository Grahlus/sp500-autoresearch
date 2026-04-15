import unittest

import pandas as pd

from experiment_validation_tags import summarize_holdout_checks, summarize_validation_tags


class ValidationTagTests(unittest.TestCase):
    def test_validation_tags_classify_horizon_and_regime(self):
        frame = pd.DataFrame(
            [
                {
                    "timestamp_utc": "2026-04-01T00:00:00+00:00",
                    "viable": False,
                    "beats_baseline_objective": False,
                    "beats_baseline_guardrails": False,
                    "sharpe": -0.4,
                    "calmar": -0.5,
                    "total_return": -4.0,
                    "max_drawdown": -30.0,
                    "trades_per_year": 1.0,
                    "exposure": 0.05,
                    "negative_windows": 4,
                },
                {
                    "timestamp_utc": "2026-04-02T00:00:00+00:00",
                    "viable": True,
                    "beats_baseline_objective": True,
                    "beats_baseline_guardrails": True,
                    "sharpe": 0.8,
                    "calmar": 1.1,
                    "total_return": 8.0,
                    "max_drawdown": -12.0,
                    "trades_per_year": 8.0,
                    "exposure": 0.3,
                    "negative_windows": 1,
                },
                {
                    "timestamp_utc": "2026-04-03T00:00:00+00:00",
                    "viable": True,
                    "beats_baseline_objective": True,
                    "beats_baseline_guardrails": True,
                    "sharpe": 1.2,
                    "calmar": 1.4,
                    "total_return": 12.0,
                    "max_drawdown": -10.0,
                    "trades_per_year": 10.0,
                    "exposure": 0.4,
                    "negative_windows": 0,
                },
            ]
        )

        summary = summarize_validation_tags(frame)

        self.assertIn("strong_short_horizon", summary["validation_horizon_tags"])
        self.assertIn("weak_long_horizon", summary["validation_horizon_tags"])
        self.assertIn("regime_mixed", summary["validation_regime_tags"])
        self.assertIn("validation_scope", summary)
        self.assertIn(summary["validation_scope"], {"partial", "narrow", "broad"})
        self.assertGreater(summary["validation_confidence"], 0.0)
        self.assertGreater(summary["validation_coverage"], 0.0)

    def test_validation_tags_fall_back_safely_when_history_is_sparse(self):
        summary = summarize_validation_tags(pd.DataFrame())

        self.assertEqual(summary["validation_horizon_tags"], ["horizon_unknown"])
        self.assertEqual(summary["validation_regime_tags"], ["regime_unknown"])
        self.assertEqual(summary["validation_scope"], "unknown")
        self.assertEqual(summary["validation_confidence"], 0.0)
        self.assertEqual(summary["validation_coverage"], 0.0)

    def test_holdout_checks_map_narrow_horizon_and_regime_signals(self):
        holdout = summarize_holdout_checks(
            validation_horizon_tags=["weak_long_horizon"],
            validation_regime_tags=["regime_mixed", "weak_in_high_vol"],
            validation_scope="narrow",
            validation_confidence=0.31,
            validation_coverage=0.25,
            targeted_follow_up_type="mixed_regime_clarification",
            holdout_check_status=None,
            holdout_check_outcome=None,
            promotion_state="provisional",
        )

        self.assertTrue(holdout["holdout_check_required"])
        self.assertEqual(holdout["holdout_check_type"], "mixed_regime_clarification_holdout")
        self.assertEqual(holdout["holdout_check_scope"], "mixed_regime")
        self.assertIn("holdout_long_unproven", holdout["holdout_horizon_tags"])
        self.assertIn("holdout_regime_mixed", holdout["holdout_regime_tags"])

    def test_holdout_checks_map_combined_long_horizon_and_high_volatility(self):
        holdout = summarize_holdout_checks(
            validation_horizon_tags=["weak_long_horizon"],
            validation_regime_tags=["weak_in_high_vol"],
            validation_scope="partial",
            validation_confidence=0.55,
            validation_coverage=0.60,
            targeted_follow_up_type="long_horizon_high_volatility_confirmation",
            holdout_check_status=None,
            holdout_check_outcome=None,
            promotion_state="provisional",
        )

        self.assertTrue(holdout["holdout_check_required"])
        self.assertEqual(holdout["holdout_check_type"], "long_horizon_high_volatility_holdout")
        self.assertEqual(holdout["holdout_check_scope"], "long_horizon_high_volatility")
        self.assertIn("holdout_long_unproven", holdout["holdout_horizon_tags"])
        self.assertIn("holdout_high_vol_unproven", holdout["holdout_regime_tags"])

    def test_holdout_checks_mark_broad_winners_as_not_required(self):
        holdout = summarize_holdout_checks(
            validation_horizon_tags=["strong_short_horizon", "stable_medium_horizon", "strong_long_horizon"],
            validation_regime_tags=["stable_in_trend", "strong_in_bear"],
            validation_scope="broad",
            validation_confidence=0.92,
            validation_coverage=0.90,
            confirmation_outcome="passed",
            promotion_state="confirmed",
        )

        self.assertFalse(holdout["holdout_check_required"])
        self.assertEqual(holdout["holdout_check_outcome"], "confirmed")
        self.assertEqual(holdout["holdout_check_status"], "completed")


if __name__ == "__main__":
    unittest.main()
