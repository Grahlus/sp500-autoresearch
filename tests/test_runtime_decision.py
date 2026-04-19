import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd

from experiment_dashboard import BestResultsDashboard
from experiment_idea_yield import build_idea_yield_summary, save_idea_yield_summary
from experiment_runtime_decision import _confirmation_batch_plan, _targeted_follow_up_plan, build_runtime_decision, save_runtime_decision
from experiment_memory import load_research_memory
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
        self.assertIn("idea_yield_summary", decision.used_signals)
        self.assertGreater(decision.new_idea_quota or 0, 0)
        self.assertGreater(decision.uncommon_idea_quota or 0, 0)
        self.assertLessEqual(decision.uncommon_idea_quota or 0, decision.new_idea_quota or 0)
        self.assertGreater(decision.repeat_branch_cap or 0, 0)
        self.assertGreater(decision.max_same_template_per_cycle or 0, 0)
        self.assertGreater(decision.max_same_lineage_per_cycle or 0, 0)
        self.assertEqual(decision.structural_novelty_threshold, 0.55)
        self.assertIn("novelty_policy", decision.used_signals)
        self.assertEqual(decision.used_signals["novelty_policy"]["uncommon_idea_quota"], decision.uncommon_idea_quota)
        self.assertIn("family_budget_rationale", decision.rationale)
        self.assertIn("anti_overfitting", decision.rationale)
        self.assertIn("confirmation_plan", decision.rationale)
        self.assertIn("novelty_policy", decision.rationale)
        self.assertEqual(decision.rationale["novelty_policy"]["new_idea_quota"], decision.new_idea_quota)
        self.assertEqual(decision.rationale["novelty_policy"]["uncommon_idea_quota"], decision.uncommon_idea_quota)
        self.assertIn("robustness_score", decision.rationale["family_budget_rationale"]["momentum"])
        self.assertIn("overfit_risk", decision.rationale["family_budget_rationale"]["momentum"])
        self.assertEqual(decision.rationale["family_budget_rationale"]["momentum"]["overfit_risk_model"], "graded_v2")
        self.assertIn(decision.rationale["family_budget_rationale"]["momentum"]["idea_state"], {"promising", "active", "untested"})
        self.assertIn("idea_quality_score", decision.rationale["family_budget_rationale"]["momentum"])
        self.assertGreater(decision.rationale["family_budget_rationale"]["momentum"]["robustness_score"], decision.rationale["family_budget_rationale"]["superstock"]["robustness_score"])
        self.assertTrue(set(decision.winner_validation_horizon_tags or []) & {"strong_short_horizon", "stable_medium_horizon", "strong_long_horizon"})
        self.assertTrue(set(decision.winner_validation_regime_tags or []) & {"stable_in_trend", "strong_in_bear", "regime_mixed"})
        self.assertIn(decision.winner_validation_scope, {"broad", "partial"})
        self.assertFalse(decision.winner_validation_needs_follow_up)
        self.assertFalse(decision.holdout_check_required)
        self.assertEqual(decision.holdout_check_outcome, "broadly_confirmed")
        self.assertFalse(decision.targeted_follow_up_required)
        self.assertIsNone(decision.targeted_follow_up_type)
        self.assertFalse(decision.rationale["targeted_follow_up"]["required"])
        self.assertIn("holdout_check", decision.rationale)
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
        self.assertGreaterEqual(decision.family_budgets["rl_bandit"], 1)
        self.assertGreaterEqual(decision.family_budgets["ml_ranker"], 1)
        self.assertEqual(decision.rationale["family_budget_rationale"]["momentum"]["budget_stance"], "primary")
        self.assertEqual(decision.rationale["family_budget_rationale"]["superstock"]["budget_stance"], "controlled")
        self.assertEqual(decision.rationale["family_budget_rationale"]["rl_bandit"]["budget_stance"], "controlled")
        self.assertEqual(decision.rationale["family_budget_rationale"]["ml_ranker"]["budget_stance"], "controlled")
        self.assertIn("promoted to primary runtime focus", decision.rationale["family_budget_rationale"]["momentum"]["reason"])
        self.assertIn("heavily reduced", decision.rationale["family_budget_rationale"]["superstock"]["reason"])
        self.assertIn("ml_", decision.rationale["family_budget_rationale"]["ml_ranker"]["focus_mode"])
        self.assertIn("rl_", decision.rationale["family_budget_rationale"]["rl_bandit"]["focus_mode"])

    def test_runtime_budget_changes_with_structural_family_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="m1",
                family="momentum",
                objective_score=1.2,
                viable=True,
                beats_baseline=True,
                config={"LOOKBACK_WEEKS": 26},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="s1",
                family="superstock",
                objective_score=1.2,
                viable=True,
                beats_baseline=True,
                config={"max_positions": 5},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="r1",
                family="rl_bandit",
                objective_score=1.0,
                viable=True,
                beats_baseline=True,
                config={"policy_type": "ucb"},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="mrank1",
                family="ml_ranker",
                objective_score=1.0,
                viable=True,
                beats_baseline=True,
                config={"model_type": "ridge"},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="m_structural",
                family="momentum",
                objective_score=0.1,
                viable=False,
                beats_baseline=False,
                config={"LOOKBACK_WEEKS": 30},
            )

            summary = build_idea_yield_summary(families=["momentum", "superstock"], base_dir=str(experiments_dir), persist_memory=False)
            momentum_summary = summary["families"]["momentum"]
            momentum_structural = momentum_summary["records"]["synthesized_template_family"]["momentum::synthesized_template_family::unknown"]
            momentum_summary.update(
                {
                    "structural_family_state": "untested",
                    "structural_family_state_previous": None,
                    "structural_family_promotion_state": "floor_protected",
                    "structural_family_transition_reason": "no execution evidence yet",
                    "graduated_template_family": None,
                    "structural_family_feedback": {
                        "old_state": None,
                        "new_state": "untested",
                        "promotion_state": "floor_protected",
                        "reason": "no execution evidence yet",
                        "graduated_template_family": None,
                    },
                }
            )
            momentum_structural.update(
                {
                    "structural_family_state": "untested",
                    "structural_family_state_previous": None,
                    "structural_family_promotion_state": "floor_protected",
                    "structural_family_transition_reason": "no execution evidence yet",
                    "graduated_template_family": None,
                }
            )
            if momentum_summary.get("top_synthesized_template_families"):
                momentum_summary["top_synthesized_template_families"][0].update(momentum_structural)
            save_idea_yield_summary(summary, base_dir=str(experiments_dir))

            floor_decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock", "rl_bandit", "ml_ranker"],
                    max_experiments=16,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                )
            )

            momentum_summary.update(
                {
                    "structural_family_state": "promising",
                    "structural_family_state_previous": "untested",
                    "structural_family_promotion_state": "graduated_structural_family",
                    "structural_family_transition_reason": "positive viable and baseline-beating evidence is durable enough to graduate",
                    "graduated_template_family": "portfolio_overlay",
                    "structural_family_feedback": {
                        "old_state": "untested",
                        "new_state": "promising",
                        "promotion_state": "graduated_structural_family",
                        "reason": "positive viable and baseline-beating evidence is durable enough to graduate",
                        "graduated_template_family": "portfolio_overlay",
                    },
                }
            )
            momentum_structural.update(
                {
                    "structural_family_state": "promising",
                    "structural_family_state_previous": "untested",
                    "structural_family_promotion_state": "graduated_structural_family",
                    "structural_family_transition_reason": "positive viable and baseline-beating evidence is durable enough to graduate",
                    "graduated_template_family": "portfolio_overlay",
                }
            )
            if momentum_summary.get("top_synthesized_template_families"):
                momentum_summary["top_synthesized_template_families"][0].update(momentum_structural)
            save_idea_yield_summary(summary, base_dir=str(experiments_dir))

            promoted_decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock", "rl_bandit", "ml_ranker"],
                    max_experiments=16,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                )
            )

        self.assertLess(floor_decision.family_budgets["momentum"], promoted_decision.family_budgets["momentum"])
        self.assertEqual(floor_decision.rationale["family_budget_rationale"]["momentum"]["structural_family_promotion_state"], "floor_protected")
        self.assertEqual(promoted_decision.rationale["family_budget_rationale"]["momentum"]["structural_family_promotion_state"], "graduated_structural_family")
        self.assertIn("structural_state=", promoted_decision.rationale["family_budget_rationale"]["momentum"]["reason"])
        self.assertIn("->graduated_structural_family", promoted_decision.rationale["family_budget_rationale"]["momentum"]["reason"])
        self.assertIn("idea_yield_summary", promoted_decision.used_signals)

    def test_runtime_decision_allows_small_ml_rl_probe_when_momentum_is_stalled(self):
        fake_dashboard = BestResultsDashboard(
            generated_at_utc="2026-04-13T12:00:00+00:00",
            base_dir="/tmp/experiments",
            ranking_policy="viable_first",
            counts={"official_result_rows": 120},
            top_overall=[
                {
                    "strategy_family": "momentum",
                    "config_hash": "m1",
                    "objective_score": 1.8,
                    "viable": True,
                    "sharpe": 1.05,
                    "calmar": 2.4,
                    "total_return": 13.2,
                    "max_drawdown": -12.0,
                    "trades_per_year": 28.0,
                    "exposure": 0.55,
                    "beats_baseline_objective": True,
                    "beats_baseline_guardrails": True,
                }
            ],
            top_viable=[
                {
                    "strategy_family": "momentum",
                    "config_hash": "m1",
                    "objective_score": 1.8,
                    "viable": True,
                    "sharpe": 1.05,
                    "calmar": 2.4,
                    "total_return": 13.2,
                    "max_drawdown": -12.0,
                    "trades_per_year": 28.0,
                    "exposure": 0.55,
                    "beats_baseline_objective": True,
                    "beats_baseline_guardrails": True,
                }
            ],
            top_baseline_beating=[
                {
                    "strategy_family": "momentum",
                    "config_hash": "m1",
                    "objective_score": 1.8,
                    "viable": True,
                    "sharpe": 1.05,
                    "calmar": 2.4,
                    "total_return": 13.2,
                    "max_drawdown": -12.0,
                    "trades_per_year": 28.0,
                    "exposure": 0.55,
                    "beats_baseline_objective": True,
                    "beats_baseline_guardrails": True,
                }
            ],
            top_per_family={
                "momentum": [
                    {
                        "strategy_family": "momentum",
                        "config_hash": "m1",
                        "objective_score": 1.8,
                        "viable": True,
                        "sharpe": 1.05,
                        "calmar": 2.4,
                        "total_return": 13.2,
                        "max_drawdown": -12.0,
                        "trades_per_year": 28.0,
                        "exposure": 0.55,
                        "beats_baseline_objective": True,
                        "beats_baseline_guardrails": True,
                    }
                ],
                "superstock": [{"strategy_family": "superstock", "config_hash": "s1", "objective_score": 0.2, "viable": False}],
                "ml_ranker": [{"strategy_family": "ml_ranker", "config_hash": "ml1", "objective_score": -0.4, "viable": False}],
                "rl_bandit": [{"strategy_family": "rl_bandit", "config_hash": "rl1", "objective_score": -0.6, "viable": False}],
            },
            latest_non_empty_batch={"batch_id": "batch_recent", "executed_count": 24},
            family_scorecards={
                "momentum": {
                    "stagnation_experiments": 18,
                    "recent_viable_trend": 0.0,
                    "recent_objective_trend": 0.0,
                    "robustness_score": 0.89,
                    "overfit_risk": 0.14,
                    "viable_rate": 0.78,
                    "win_rate_vs_baseline": 0.64,
                    "validation_scope": "broad",
                    "validation_confidence": 0.95,
                    "validation_coverage": 0.95,
                    "validation_horizon_tags": ["strong_short_horizon", "stable_medium_horizon", "strong_long_horizon"],
                    "validation_regime_tags": ["stable_in_trend", "strong_in_bear"],
                    "overfit_risk_model": "graded_v2",
                    "promotion_state": "confirmed",
                    "winner_promotion_status": "promoted",
                    "holdout_check_type": "long_horizon_high_volatility_holdout",
                    "holdout_check_status": "completed",
                    "holdout_check_outcome": "confirmed",
                    "holdout_check_scope": "long_horizon_high_volatility",
                    "holdout_horizon_tags": ["holdout_long_confirmed"],
                    "holdout_regime_tags": ["holdout_high_vol_confirmed"],
                },
                "superstock": {
                    "stagnation_experiments": 4,
                    "recent_viable_trend": -0.04,
                    "recent_objective_trend": -0.03,
                    "robustness_score": 0.22,
                    "overfit_risk": 0.88,
                    "viable_rate": 0.02,
                    "win_rate_vs_baseline": 0.0,
                    "validation_scope": "narrow",
                    "validation_horizon_tags": ["weak_long_horizon"],
                    "validation_regime_tags": ["weak_in_high_vol"],
                    "overfit_risk_model": "graded_v2",
                },
                "ml_ranker": {
                    "stagnation_experiments": 22,
                    "recent_viable_trend": 0.0,
                    "recent_objective_trend": -0.05,
                    "robustness_score": 0.28,
                    "overfit_risk": 0.69,
                    "viable_rate": 0.0,
                    "win_rate_vs_baseline": 0.0,
                    "validation_scope": "partial",
                    "validation_horizon_tags": ["weak_long_horizon"],
                    "validation_regime_tags": ["weak_in_high_vol"],
                    "overfit_risk_model": "graded_v2",
                },
                "rl_bandit": {
                    "stagnation_experiments": 24,
                    "recent_viable_trend": 0.0,
                    "recent_objective_trend": -0.08,
                    "robustness_score": 0.24,
                    "overfit_risk": 0.72,
                    "viable_rate": 0.0,
                    "win_rate_vs_baseline": 0.0,
                    "validation_scope": "partial",
                    "validation_horizon_tags": ["weak_long_horizon"],
                    "validation_regime_tags": ["weak_in_high_vol"],
                    "overfit_risk_model": "graded_v2",
                },
            },
            lineage_summary={"latest_batch_id": "batch_recent", "lineage_status_counts": {}, "family_summaries": {}},
        )

        with patch("experiment_runtime_decision.build_best_results_dashboard", return_value=fake_dashboard), patch(
            "experiment_runtime_decision.load_research_memory", return_value={}
        ):
            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root="/tmp",
                    experiments_dir="/tmp/experiments",
                    strategy_families=["momentum", "superstock", "ml_ranker", "rl_bandit"],
                    max_experiments=24,
                    exploration_fraction=0.75,
                    exploitation_fraction=0.25,
                    min_large_search_candidates=48,
                )
            )

        self.assertEqual(decision.winner_family, "momentum")
        self.assertEqual(decision.family_budgets["momentum"], max(decision.family_budgets.values()))
        self.assertGreaterEqual(decision.family_budgets["ml_ranker"], 1)
        self.assertGreaterEqual(decision.family_budgets["rl_bandit"], 1)
        self.assertEqual(decision.rationale["family_budget_rationale"]["ml_ranker"]["budget_stance"], "controlled")
        self.assertEqual(decision.rationale["family_budget_rationale"]["rl_bandit"]["budget_stance"], "controlled")
        self.assertIn(decision.rationale["family_budget_rationale"]["ml_ranker"]["focus_mode"], {"ml_rethink_probe", "ml_small_exploration"})
        self.assertIn(decision.rationale["family_budget_rationale"]["rl_bandit"]["focus_mode"], {"rl_rethink_probe", "rl_small_exploration"})
        self.assertIn("ml_rl_probe_policy", decision.rationale)
        self.assertTrue(decision.rationale["ml_rl_probe_policy"]["allowed"])
        self.assertFalse(decision.rationale["ml_rl_probe_policy"]["non_momentum_viable_progress"])

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
        self.assertTrue(
            "weak_long_horizon" in decision.winner_validation_horizon_tags
            or "horizon_mixed" in decision.winner_validation_horizon_tags
        )
        self.assertTrue(decision.confirmation_required or decision.winner_promotion_status in {"hold_for_confirmation", "cautious_promotion"})
        self.assertLessEqual(decision.winner_exploitation_cap, 0.4)
        self.assertIn(decision.rationale["winner_promotion_policy"]["signals"]["validation_scope"], {"narrow", "partial"})
        self.assertTrue(decision.targeted_follow_up_required)
        self.assertIsNotNone(decision.targeted_follow_up_type)
        self.assertGreater(decision.targeted_follow_up_priority or 0.0, 0.0)
        self.assertTrue(decision.rationale["targeted_follow_up"]["required"])
        self.assertIn(
            decision.rationale["targeted_follow_up"]["type"],
            {
                "long_horizon_confirmation",
                "coverage_expansion_confirmation",
                "mixed_regime_clarification",
                "long_horizon_high_volatility_confirmation",
                "targeted_follow_up_confirmation",
            },
        )
        self.assertIn("reason", decision.rationale["targeted_follow_up"])
        self.assertTrue(decision.holdout_check_required)
        self.assertIn(
            decision.holdout_check_type,
            {
                "long_horizon_holdout",
                "coverage_expansion_holdout",
                "mixed_regime_clarification_holdout",
                "generic_holdout",
                "long_horizon_high_volatility_holdout",
            },
        )

    def test_targeted_follow_up_plan_maps_mixed_regime_to_clarification(self):
        plan = _targeted_follow_up_plan(
            family="momentum",
            scorecard={
                "validation_horizon_tags": ["stable_medium_horizon"],
                "validation_regime_tags": ["regime_mixed"],
                "validation_scope": "partial",
                "validation_confidence": 0.72,
                "validation_coverage": 0.55,
                "robustness_score": 0.58,
                "overfit_risk": 0.42,
                "recent_robustness_trend": -0.02,
                "viable_rate": 0.18,
                "win_rate_vs_baseline": 0.46,
            },
            promotion_policy={"winner_promotion_status": "cautious_promotion"},
        )

        self.assertTrue(plan["targeted_follow_up_required"])
        self.assertEqual(plan["targeted_follow_up_type"], "mixed_regime_clarification")
        self.assertGreater(plan["targeted_follow_up_priority"], 0.0)
        self.assertIn("regime evidence is mixed", plan["targeted_follow_up_reason"])

    def test_targeted_follow_up_plan_prefers_combined_long_and_high_vol_holdout(self):
        plan = _targeted_follow_up_plan(
            family="momentum",
            scorecard={
                "validation_horizon_tags": ["weak_long_horizon"],
                "validation_regime_tags": ["weak_in_high_vol"],
                "validation_scope": "partial",
                "validation_confidence": 0.61,
                "validation_coverage": 0.64,
                "robustness_score": 0.54,
                "overfit_risk": 0.58,
                "overfit_risk_model": "graded_v2",
                "recent_robustness_trend": -0.03,
                "viable_rate": 0.12,
                "win_rate_vs_baseline": 0.41,
                "lineage_status_summary": "leaf",
                "lineage_trust_score": 0.20,
            },
            promotion_policy={"winner_promotion_status": "hold_for_confirmation"},
        )

        self.assertTrue(plan["targeted_follow_up_required"])
        self.assertEqual(plan["targeted_follow_up_type"], "long_horizon_high_volatility_confirmation")
        self.assertIn("both long-horizon and high-volatility validation are weak", plan["targeted_follow_up_reason"])
        self.assertGreater(plan["targeted_follow_up_priority"], 0.0)

    def test_confirmation_batch_plan_assigns_dedicated_combined_holdout_batch(self):
        plan = _confirmation_batch_plan(
            decision_id="runtime_20260413_123456",
            request=RuntimeDecisionInput(
                workspace_root="/tmp",
                experiments_dir="/tmp/experiments",
                strategy_families=["momentum"],
                max_experiments=24,
                exploration_fraction=0.65,
                exploitation_fraction=0.35,
                min_large_search_candidates=48,
            ),
            winner_family="momentum",
            winner_scorecard={
                "validation_horizon_tags": ["weak_long_horizon"],
                "validation_regime_tags": ["weak_in_high_vol"],
                "validation_scope": "partial",
                "validation_confidence": 0.61,
                "validation_coverage": 0.64,
                "robustness_score": 0.54,
                "overfit_risk": 0.58,
                "overfit_risk_model": "graded_v2",
                "recent_robustness_trend": -0.03,
                "viable_rate": 0.12,
                "win_rate_vs_baseline": 0.41,
            },
            promotion_policy={
                "winner_family": "momentum",
                "winner_promotion_status": "hold_for_confirmation",
                "winner_exploitation_cap": 0.25,
                "confirmation_batch_requested": True,
            },
            latest_batch_overview=None,
        )

        self.assertTrue(plan["holdout_check_required"])
        self.assertEqual(plan["holdout_check_type"], "long_horizon_high_volatility_holdout")
        self.assertEqual(plan["holdout_check_scope"], "long_horizon_high_volatility")
        self.assertEqual(plan["holdout_check_batch_id"], "runtime_20260413_123456_holdout_momentum")
        self.assertTrue(plan["confirmation_required"])

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
        self.assertEqual(decision.rationale["anti_overfitting"]["family_risk_reports"]["momentum"]["overfit_risk_model"], "graded_v2")
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
        self.assertFalse(decision.targeted_follow_up_required)

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

    def test_runtime_decision_persists_confirmed_promotion_state_across_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="momentum_confirmed",
                family="momentum",
                objective_score=1.55,
                viable=True,
                beats_baseline=True,
                config={"LOOKBACK_WEEKS": 24, "FG_MIN": 10.0},
            )
            _save_result(
                str(experiments_dir),
                experiment_id="superstock_control",
                family="superstock",
                objective_score=0.35,
                viable=False,
                beats_baseline=False,
                config={"max_positions": 5},
            )
            batch_dir = experiments_dir / "batches" / "proposal_20260413_confirmed_batch"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "batch_id": "proposal_20260413_confirmed_batch",
                        "timestamp_utc": "2026-04-13T00:00:00+00:00",
                        "total_requested": 6,
                        "total_sampled": 6,
                        "total_executed": 6,
                        "total_failed": 0,
                        "shortfall": 0,
                        "status_counts": {"success": 6},
                        "proposal_metadata": {
                            "confirmation_required": True,
                            "confirmation_state": "unconfirmed",
                            "confirmation_batch_id": "runtime_123_confirm_momentum",
                            "confirmation_outcome": "passed",
                        },
                    }
                )
            )

            decision1 = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock"],
                    max_experiments=24,
                )
            )
            save_runtime_decision(decision1, workspace_root=tmp)
            memory = load_research_memory(str(experiments_dir))
            record = (memory.get("promotion_states") or {}).get("momentum", {}).get("momentum_confirmed")

            decision2 = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum", "superstock"],
                    max_experiments=24,
                )
            )

        self.assertIsInstance(record, dict)
        self.assertEqual(record["promotion_state"], "confirmed")
        self.assertEqual(decision2.promotion_state, "confirmed")
        self.assertFalse(decision2.promotion_state_blocked_pending_new_evidence)
        self.assertEqual(decision2.promotion_state_record["promotion_state"], "confirmed")
        self.assertEqual(decision2.used_signals["promotion_state_record"]["promotion_state"], "confirmed")
        self.assertIn("persisted_promotion_state", decision2.used_signals)
        self.assertEqual(decision2.family_scorecards["momentum"]["promotion_state"], "confirmed")
        self.assertGreaterEqual(decision2.winner_exploitation_cap or 0.0, 0.50)
        self.assertIn("persisted confirmation state", decision2.rationale["winner_promotion_policy"]["reasons"][0])

    def test_runtime_decision_reuses_blocked_promotion_state_on_followup_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="momentum_blocked",
                family="momentum",
                objective_score=1.55,
                viable=True,
                beats_baseline=True,
                config={"LOOKBACK_WEEKS": 24, "FG_MIN": 10.0},
            )
            batch_dir = experiments_dir / "batches" / "proposal_20260413_confirm_batch"
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "batch_id": "proposal_20260413_confirm_batch",
                        "timestamp_utc": "2026-04-13T00:00:00+00:00",
                        "total_requested": 6,
                        "total_sampled": 1,
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

            decision1 = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=24,
                )
            )
            save_runtime_decision(decision1, workspace_root=tmp)
            decision2 = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=24,
                )
            )

        self.assertEqual(decision1.confirmation_outcome, "failed")
        self.assertEqual(decision1.promotion_state, "rejected")
        self.assertIn(decision2.promotion_state, {"rejected", "blocked_pending_new_evidence"})
        self.assertTrue(decision2.promotion_state_blocked_pending_new_evidence)
        self.assertEqual(decision2.cycle_mode, "diagnostics")
        self.assertTrue(decision2.rationale["promotion_state_blocked_pending_new_evidence"])
        self.assertIn("blocked pending new evidence", " ".join(decision2.rationale["winner_promotion_policy"]["reasons"]))


if __name__ == "__main__":
    unittest.main()
