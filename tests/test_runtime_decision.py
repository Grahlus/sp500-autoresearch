import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_runtime_decision import build_runtime_decision, save_runtime_decision
from experiment_store import init_store, save_experiment_result
from experiment_types import RuntimeDecisionInput


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    objective_score: float,
    viable: bool,
    beats_baseline: bool,
    config: dict[str, Any] | None = None,
) -> None:
    params_by_family = {
        "momentum": config or {"LOOKBACK_WEEKS": 26},
        "superstock": config or {"max_positions": 5},
        "ml_ranker": config or {"model_type": "ridge"},
        "rl_bandit": config or {"policy_type": "ucb"},
    }
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": params_by_family.get(family, config or {"LOOKBACK_WEEKS": 26}),
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "runtime_test",
                "config_hash": experiment_id,
                "experiment_id": experiment_id,
                "timestamp_utc": "2026-04-12T00:00:00+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-12",
                "split": "walk-forward",
            },
            "status": "success",
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
                "exposure": 0.5 if viable else 0.05,
            },
            "robustness": {"negative_windows": 0 if viable else 3, "viable": viable},
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "baseline_verified": True,
                "baseline_metric_source": "verified_current_engine",
                "comparison_kind": "partial",
                "delta_sharpe": objective_score - 1.0,
                "delta_calmar": objective_score - 1.0,
                "delta_return": objective_score * 10.0,
                "beats_baseline_objective": beats_baseline,
                "beats_baseline_guardrails": beats_baseline,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class RuntimeDecisionTests(unittest.TestCase):
    def test_runtime_decision_uses_dashboard_evidence_for_active_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            for idx in range(40):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"m_extra_{idx}",
                    family="momentum",
                    objective_score=2.0,
                    viable=True,
                    beats_baseline=True,
                    config={"LOOKBACK_WEEKS": 20 + idx},
                )
            _save_result(str(experiments_dir), experiment_id="m1", family="momentum", objective_score=2.0, viable=True, beats_baseline=True)
            _save_result(str(experiments_dir), experiment_id="s1", family="superstock", objective_score=0.4, viable=False, beats_baseline=False)

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock"],
                    max_experiments=24,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    min_large_search_candidates=48,
                )
            )
            report_path = save_runtime_decision(decision, workspace_root=tmp)
            payload = json.loads(report_path.read_text())

        self.assertEqual(decision.status, "active")
        self.assertEqual(decision.cycle_mode, "local_refinement")
        self.assertGreater(decision.family_budgets["momentum"], decision.family_budgets["superstock"])
        self.assertEqual(decision.best_viable["strategy_family"], "momentum")
        self.assertEqual(decision.best_baseline_beating["strategy_family"], "momentum")
        self.assertEqual(decision.winner_family, "momentum")
        self.assertEqual(decision.promotion_state, "confirmed")
        self.assertEqual(decision.winner_promotion_status, "promoted")
        self.assertFalse(decision.confirmation_batch_requested)
        self.assertFalse(decision.confirmation_required)
        self.assertGreaterEqual(decision.winner_exploitation_cap, 0.55)
        self.assertIn("best_viable", decision.used_signals)
        self.assertIn("family_budget_rationale", decision.rationale)
        self.assertIn("anti_overfitting", decision.rationale)
        self.assertIn("confirmation_plan", decision.rationale)
        self.assertIn("robustness_score", decision.rationale["family_budget_rationale"]["momentum"])
        self.assertIn("overfit_risk", decision.rationale["family_budget_rationale"]["momentum"])
        self.assertGreater(decision.rationale["family_budget_rationale"]["momentum"]["robustness_score"], decision.rationale["family_budget_rationale"]["superstock"]["robustness_score"])
        self.assertTrue(set(decision.winner_validation_horizon_tags or []) & {"strong_short_horizon", "stable_medium_horizon", "strong_long_horizon"})
        self.assertTrue(set(decision.winner_validation_regime_tags or []) & {"stable_in_trend", "strong_in_bear", "regime_mixed"})
        self.assertIn(decision.winner_validation_scope, {"broad", "partial"})
        self.assertFalse(decision.winner_validation_needs_follow_up)
        self.assertIn("runtime_", payload["decision_id"])
        self.assertIn("runtime decision", payload["rationale"]["reason"].lower())
        self.assertEqual(payload["winner_promotion_status"], "promoted")
        self.assertEqual(payload["promotion_state"], "confirmed")
        self.assertIn("winner_promotion_policy", payload["rationale"])

    def test_runtime_decision_rebalances_families_from_dashboard_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))

            for idx in range(12):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"momentum_good_{idx}",
                    family="momentum",
                    objective_score=1.8,
                    viable=True,
                    beats_baseline=True,
                    config={"LOOKBACK_WEEKS": 20 + idx, "FG_MIN": 10.0},
                )
            for idx in range(8):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"superstock_fragile_{idx}",
                    family="superstock",
                    objective_score=2.4,
                    viable=False,
                    beats_baseline=False,
                    config={"max_positions": 3 + idx % 2, "lookback_days": 126},
                )
            for idx in range(6):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"rl_failed_{idx}",
                    family="rl_bandit",
                    objective_score=-0.5,
                    viable=False,
                    beats_baseline=False,
                    config={"policy_type": "ucb", "lookback_days": 126 + idx},
                )
            for idx in range(6):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"ml_failed_{idx}",
                    family="ml_ranker",
                    objective_score=-0.7,
                    viable=False,
                    beats_baseline=False,
                    config={"model_type": "ridge", "lookback_days": 126 + idx},
                )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock", "rl_bandit", "ml_ranker"],
                    max_experiments=24,
                    exploration_fraction=0.75,
                    exploitation_fraction=0.25,
                    min_large_search_candidates=48,
                )
            )

        self.assertEqual(decision.family_budgets["momentum"], max(decision.family_budgets.values()))
        self.assertGreater(decision.family_budgets["momentum"], decision.family_budgets["superstock"])
        self.assertGreaterEqual(decision.family_budgets["superstock"], 1)
        self.assertEqual(decision.family_budgets["rl_bandit"], 0)
        self.assertEqual(decision.family_budgets["ml_ranker"], 0)
        self.assertEqual(decision.rationale["family_budget_rationale"]["momentum"]["budget_stance"], "primary")
        self.assertEqual(decision.rationale["family_budget_rationale"]["superstock"]["budget_stance"], "controlled")
        self.assertEqual(decision.rationale["family_budget_rationale"]["rl_bandit"]["budget_stance"], "paused")
        self.assertEqual(decision.rationale["family_budget_rationale"]["ml_ranker"]["budget_stance"], "paused")
        self.assertIn("promoted to primary runtime focus", decision.rationale["family_budget_rationale"]["momentum"]["reason"])
        self.assertIn("heavily reduced", decision.rationale["family_budget_rationale"]["superstock"]["reason"])
        self.assertIn("temporarily paused", decision.rationale["family_budget_rationale"]["rl_bandit"]["reason"])
        self.assertIn("temporarily paused", decision.rationale["family_budget_rationale"]["ml_ranker"]["reason"])

    def test_runtime_decision_restricts_narrow_validation_winners(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="momentum_old_weak",
                family="momentum",
                objective_score=-0.2,
                viable=False,
                beats_baseline=False,
                config={"LOOKBACK_WEEKS": 18},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="momentum_mid_weak",
                family="momentum",
                objective_score=0.1,
                viable=False,
                beats_baseline=False,
                config={"LOOKBACK_WEEKS": 20},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="momentum_recent_strong",
                family="momentum",
                objective_score=1.7,
                viable=True,
                beats_baseline=True,
                config={"LOOKBACK_WEEKS": 22},
            )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=24,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    min_large_search_candidates=48,
                )
            )

        self.assertIn(decision.winner_validation_scope, {"narrow", "partial"})
        self.assertTrue(decision.winner_validation_needs_follow_up)
        self.assertIn("weak_long_horizon", decision.winner_validation_horizon_tags)
        self.assertTrue(decision.confirmation_required or decision.winner_promotion_status in {"hold_for_confirmation", "cautious_promotion"})
        self.assertLessEqual(decision.winner_exploitation_cap, 0.4)
        self.assertIn(decision.rationale["winner_promotion_policy"]["signals"]["validation_scope"], {"narrow", "partial"})

    def test_runtime_decision_downweights_suspicious_overfit_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            save_experiment_result(
                {
                    "spec": {
                        "family": "momentum",
                        "params": {"X": 1},
                        "search_method": "single",
                        "objective_name": "wf_v1_score",
                        "batch_id": "runtime_test",
                        "config_hash": "m_overfit",
                        "experiment_id": "m_overfit",
                        "timestamp_utc": "2026-04-12T00:00:00+00:00",
                        "benchmark_source": "spy_symbol",
                        "dataset_id": "data123",
                        "data_start": "2020-01-01",
                        "data_end": "2026-04-12",
                        "split": "walk-forward",
                    },
                    "status": "success",
                    "objective_score": 2.5,
                    "metrics": {
                        "sharpe": 2.5,
                        "calmar": 2.5,
                        "total_return": 25.0,
                        "max_drawdown": -35.0,
                        "trades_per_year": 0.5,
                        "exposure": 0.02,
                    },
                    "robustness": {"negative_windows": 5, "viable": False},
                    "baseline_comparison": {
                        "baseline_name": "momentum_champion_s10005",
                        "comparison_status": "partial_verified_current_engine",
                        "baseline_verified": True,
                        "baseline_metric_source": "verified_current_engine",
                        "comparison_kind": "partial",
                        "delta_sharpe": 1.5,
                        "delta_calmar": 1.5,
                        "delta_return": 15.0,
                        "beats_baseline_objective": False,
                        "beats_baseline_guardrails": False,
                    },
                    "artifacts": {},
                    "runtime_seconds": 0.1,
                },
                base_dir=str(experiments_dir),
            )
            save_experiment_result(
                {
                    "spec": {
                        "family": "superstock",
                        "params": {"X": 1},
                        "search_method": "single",
                        "objective_name": "wf_v1_score",
                        "batch_id": "runtime_test",
                        "config_hash": "s1",
                        "experiment_id": "s1",
                        "timestamp_utc": "2026-04-12T00:00:00+00:00",
                        "benchmark_source": "spy_symbol",
                        "dataset_id": "data123",
                        "data_start": "2020-01-01",
                        "data_end": "2026-04-12",
                        "split": "walk-forward",
                    },
                    "status": "success",
                    "objective_score": 1.0,
                    "metrics": {
                        "sharpe": 0.8,
                        "calmar": 1.0,
                        "total_return": 10.0,
                        "max_drawdown": -10.0,
                        "trades_per_year": 10.0,
                        "exposure": 0.4,
                    },
                    "robustness": {"negative_windows": 0, "viable": True},
                    "baseline_comparison": {
                        "baseline_name": "momentum_champion_s10005",
                        "comparison_status": "partial_verified_current_engine",
                        "baseline_verified": True,
                        "baseline_metric_source": "verified_current_engine",
                        "comparison_kind": "partial",
                        "delta_sharpe": 0.0,
                        "delta_calmar": 0.0,
                        "delta_return": 10.0,
                        "beats_baseline_objective": False,
                        "beats_baseline_guardrails": False,
                    },
                    "artifacts": {},
                    "runtime_seconds": 0.1,
                },
                base_dir=str(experiments_dir),
            )
            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock"],
                    max_experiments=24,
                    exploration_fraction=0.50,
                    exploitation_fraction=0.50,
                    min_large_search_candidates=48,
                )
            )

        self.assertEqual(decision.status, "active")
        self.assertEqual(decision.cycle_mode, "confirmation")
        self.assertGreaterEqual(decision.exploration_fraction, 0.75)
        self.assertTrue(decision.confirmation_batch_requested)
        self.assertTrue(decision.confirmation_required)
        self.assertEqual(decision.winner_promotion_status, "hold_for_confirmation")
        self.assertEqual(decision.promotion_state, "unconfirmed")
        self.assertLessEqual(decision.winner_exploitation_cap, 0.25)
        self.assertLess(decision.family_budgets["momentum"], 18)
        self.assertGreater(decision.exploration_fraction, 0.60)
        self.assertTrue(decision.rationale["anti_overfitting"]["do_not_over_exploit_yet"])
        self.assertIn("confirmation_reason", decision.rationale)
        self.assertIn("confirmation_batch_id", decision.rationale)
        self.assertIn("planned_max_experiments", decision.rationale)
        self.assertGreater(decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["risk_score"], 0.45)
        self.assertGreater(decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["scorecard_overfit_risk"], 0.5)
        self.assertIn("robustness_score", decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["signals"])
        self.assertIn("not_viable", decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["flags"])
        self.assertIn("high_objective_non_viable", decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["flags"])
        self.assertEqual(decision.rationale["winner_promotion_policy"]["winner_promotion_status"], "hold_for_confirmation")
        self.assertTrue(decision.rationale["winner_promotion_policy"]["confirmation_batch_requested"])

    def test_runtime_decision_marks_rejected_confirmation_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(str(experiments_dir), experiment_id="m1", family="momentum", objective_score=1.6, viable=True, beats_baseline=True)
            batch_dir = experiments_dir / "batches" / "proposal_20260412_confirm_batch"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "batch_id": "proposal_20260412_confirm_batch",
                        "timestamp_utc": "2026-04-12T00:00:00+00:00",
                        "total_requested": 6,
                        "total_sampled": 6,
                        "total_executed": 0,
                        "total_failed": 0,
                        "shortfall": 6,
                        "status_counts": {},
                        "proposal_metadata": {
                            "confirmation_required": True,
                            "confirmation_state": "unconfirmed",
                            "confirmation_batch_id": "runtime_123_confirm_momentum",
                        },
                    }
                )
            )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=24,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    min_large_search_candidates=48,
                )
            )

        self.assertEqual(decision.promotion_state, "rejected")
        self.assertEqual(decision.confirmation_outcome, "failed")
        self.assertEqual(decision.cycle_mode, "diagnostics")
        self.assertFalse(decision.confirmation_required)
        self.assertFalse(decision.confirmation_batch_requested)

    def test_runtime_decision_widens_for_underfilled_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            for idx in range(40):
                _save_result(
                    str(experiments_dir),
                    experiment_id=f"m_extra_underfilled_{idx}",
                    family="momentum",
                    objective_score=1.3,
                    viable=True,
                    beats_baseline=True,
                    config={"LOOKBACK_WEEKS": 30 + idx},
                )
            _save_result(str(experiments_dir), experiment_id="m1", family="momentum", objective_score=1.2, viable=True, beats_baseline=False)
            batch_dir = experiments_dir / "batches" / "batch_20260412_underfilled"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "total_requested": 24,
                        "total_sampled": 0,
                        "total_executed": 0,
                        "shortfall": 24,
                        "status_counts": {},
                    }
                )
            )
            pd.DataFrame(columns=["experiment_id", "status"]).to_csv(batch_dir / "leaderboard.csv", index=False)

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=24,
                    exploration_fraction=0.55,
                    exploitation_fraction=0.45,
                    min_large_search_candidates=48,
                )
            )

        self.assertEqual(decision.status, "active")
        self.assertEqual(decision.cycle_mode, "diagnostics")
        self.assertGreaterEqual(decision.exploration_fraction, 0.75)
        self.assertTrue(decision.large_search_mode)
        self.assertEqual(decision.latest_batch_overview["requested_count"], 24)
        self.assertEqual(decision.latest_batch_overview["shortfall"], 24)

    def test_runtime_decision_falls_back_safely_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(Path(tmp) / "experiments"),
                    strategy_families=["momentum"],
                    max_experiments=24,
                )
            )

        self.assertEqual(decision.status, "fallback")
        self.assertEqual(decision.cycle_mode, "legacy_fallback")
        self.assertIsNone(decision.family_budgets)


if __name__ == "__main__":
    unittest.main()
