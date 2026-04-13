import json
import tempfile
import unittest
from pathlib import Path

from experiment_refinement import build_proposal_request, generate_next_round_proposal, negotiate_family_budgets
from experiment_scorecards import build_family_scorecards, save_family_scorecards
from experiment_store import save_experiment_result


def _params_for_family(family: str) -> dict:
    if family == "momentum":
        return {
            "LOOKBACK_WEEKS": 26,
            "SKIP_WEEKS": 3,
            "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025,
            "MA_WEEKS": 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.2,
            "STOP_PARABOLIC": 0.3,
            "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5,
            "FG_MIN": 10.0,
            "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None,
        }
    return {}


def _save_result(
    base_dir: str,
    *,
    family: str,
    experiment_id: str,
    config_hash: str,
    objective_score: float,
    viable: bool,
    timestamp_utc: str,
    status: str = "success",
    beats_baseline: bool | None = None,
    dead_zone_risk: float | None = None,
) -> None:
    baseline = None
    if beats_baseline is not None:
        baseline = {
            "baseline_name": "momentum_champion_s10005",
            "comparison_status": "partial_verified_current_engine",
            "beats_baseline_objective": beats_baseline,
        }
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": _params_for_family(family),
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "scorecard_test",
                "config_hash": config_hash,
                "experiment_id": experiment_id,
                "timestamp_utc": timestamp_utc,
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-03",
                "split": "walk-forward",
                "dead_zone_risk": dead_zone_risk,
            },
            "status": status,
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
            },
            "robustness": {"negative_windows": 0 if viable else 3, "viable": viable},
            "baseline_comparison": baseline,
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class ExperimentScorecardTests(unittest.TestCase):
    def test_scorecards_are_built_from_official_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_result(
                tmp,
                family="momentum",
                experiment_id="m1",
                config_hash="m1",
                objective_score=1.0,
                viable=True,
                timestamp_utc="2026-04-01T00:00:00+00:00",
                beats_baseline=True,
            )
            _save_result(
                tmp,
                family="momentum",
                experiment_id="m2",
                config_hash="m2",
                objective_score=2.0,
                viable=False,
                timestamp_utc="2026-04-02T00:00:00+00:00",
                beats_baseline=False,
                status="no_trades",
                dead_zone_risk=0.8,
            )
            scorecard = build_family_scorecards(families=["momentum"], base_dir=tmp)["momentum"]

        self.assertEqual(scorecard.family, "momentum")
        self.assertEqual(scorecard.total_experiments, 2)
        self.assertEqual(scorecard.viable_rate, 0.5)
        self.assertEqual(scorecard.win_rate_vs_baseline, 0.5)
        self.assertGreater(scorecard.robustness_score, 0.5)
        self.assertGreater(scorecard.overfit_risk, 0.5)
        self.assertIn("robustness_score", scorecard.__dict__)
        self.assertIn("overfit_risk", scorecard.__dict__)
        self.assertIn("recent_robustness_trend", scorecard.__dict__)
        self.assertIn("lineage_status_summary", scorecard.__dict__)
        self.assertIn("lineage_trust_score", scorecard.__dict__)
        self.assertIn("validation_horizon_tags", scorecard.__dict__)
        self.assertIn("validation_regime_tags", scorecard.__dict__)
        self.assertIn("validation_scope", scorecard.__dict__)
        self.assertEqual(scorecard.best_objective_score, 1.0)
        self.assertEqual(scorecard.median_objective_score, 1.5)
        self.assertGreater(scorecard.dead_zone_density, 0.0)

    def test_scorecards_surface_validation_tags_from_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_result(
                tmp,
                family="momentum",
                experiment_id="old_weak",
                config_hash="old_weak",
                objective_score=-0.2,
                viable=False,
                timestamp_utc="2026-04-01T00:00:00+00:00",
                beats_baseline=False,
                dead_zone_risk=0.8,
            )
            _save_result(
                tmp,
                family="momentum",
                experiment_id="mid_ok",
                config_hash="mid_ok",
                objective_score=0.8,
                viable=True,
                timestamp_utc="2026-04-02T00:00:00+00:00",
                beats_baseline=True,
            )
            _save_result(
                tmp,
                family="momentum",
                experiment_id="recent_strong",
                config_hash="recent_strong",
                objective_score=1.6,
                viable=True,
                timestamp_utc="2026-04-03T00:00:00+00:00",
                beats_baseline=True,
            )

            scorecard = build_family_scorecards(families=["momentum"], base_dir=tmp)["momentum"]

        self.assertIn("strong_short_horizon", scorecard.validation_horizon_tags)
        self.assertIn("weak_long_horizon", scorecard.validation_horizon_tags)
        self.assertIn("validation_scope", scorecard.__dict__)
        self.assertIn(scorecard.validation_scope, {"partial", "narrow", "broad"})
        self.assertGreater(scorecard.validation_confidence, 0.0)
        self.assertIn("lineage_status_summary", scorecard.__dict__)
        self.assertIn("lineage_trust_score", scorecard.__dict__)
        self.assertTrue(
            {"stable_in_trend", "strong_in_bear", "weak_in_high_vol"} & set(scorecard.validation_regime_tags)
        )

    def test_scorecard_persistence_writes_report_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            scorecards = build_family_scorecards(families=["momentum"], base_dir=tmp)
            path = save_family_scorecards(scorecards, base_dir=tmp, timestamp_utc="2026-04-12T00:00:00+00:00")
            payload = json.loads(path.read_text())

        self.assertEqual(path.name, "family_scorecards.json")
        self.assertIn("momentum", payload["families"])
        self.assertEqual(payload["families"]["momentum"]["total_experiments"], 0)

    def test_sparse_history_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            scorecard = build_family_scorecards(families=["ml_ranker"], base_dir=tmp)["ml_ranker"]

        self.assertEqual(scorecard.total_experiments, 0)
        self.assertEqual(scorecard.confidence, 0.0)
        self.assertEqual(scorecard.robustness_score, 0.0)
        self.assertEqual(scorecard.overfit_risk, 1.0)
        self.assertGreater(scorecard.exploration_budget_recommendation, scorecard.exploitation_budget_recommendation)

    def test_budget_negotiation_uses_scorecard_evidence(self):
        request = build_proposal_request(
            strategy_families=["momentum", "superstock"],
            max_experiments=20,
            seed=7,
            persist_scorecards=False,
        )
        analysis = {
            "families": {
                "momentum": {
                    "viable_rate": 0.0,
                    "objective_max": 0.0,
                    "objective_mean": 0.0,
                    "history_count": 100,
                    "scorecard": {"confidence": 1.0, "search_priority": 0.9, "recovery_signal": True},
                },
                "superstock": {
                    "viable_rate": 0.0,
                    "objective_max": 0.0,
                    "objective_mean": 0.0,
                    "history_count": 100,
                    "scorecard": {"confidence": 1.0, "search_priority": 0.1, "dead_zone_density": 0.8},
                },
            }
        }
        budgets, report = negotiate_family_budgets(request, analysis)

        self.assertGreater(budgets["momentum"], budgets["superstock"])
        self.assertIn("scorecard", report["family_scores"]["momentum"])

    def test_generated_proposal_records_scorecard_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            _save_result(
                tmp,
                family="momentum",
                experiment_id="m1",
                config_hash="m1",
                objective_score=1.0,
                viable=True,
                timestamp_utc="2026-04-01T00:00:00+00:00",
                beats_baseline=True,
            )
            request = build_proposal_request(
                strategy_families=["momentum"],
                max_experiments=4,
                seed=7,
                use_idea_queue=False,
                use_analysis_guidance=False,
            )
            proposal = generate_next_round_proposal(request, base_dir=tmp)
            scorecard_path = Path(proposal.reasoning_summary["family_scorecard_path"])

        self.assertIn("family_scorecards", proposal.reasoning_summary)
        self.assertTrue(scorecard_path.name.endswith("family_scorecards.json"))
        self.assertEqual(proposal.reasoning_summary["family_scorecards"]["momentum"]["total_experiments"], 1)


if __name__ == "__main__":
    unittest.main()
