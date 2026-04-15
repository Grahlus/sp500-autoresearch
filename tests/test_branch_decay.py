import tempfile
import unittest
from pathlib import Path

from experiment_lineage import build_branch_budget_plan, build_lineage_summary
from experiment_dashboard import build_best_results_dashboard
from experiment_scorecards import build_family_scorecards
from experiment_store import init_store, save_experiment_result


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    config_hash: str,
    objective_score: float,
    viable: bool,
    timestamp_utc: str,
    parent_config_hash: str | None = None,
    status: str = "success",
    exposure: float | None = None,
    trades_per_year: float | None = None,
) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"FG_MIN": 10.0},
                "search_method": "proposal",
                "objective_name": "wf_v1_score",
                "batch_id": "branch_decay_test",
                "config_hash": config_hash,
                "experiment_id": experiment_id,
                "timestamp_utc": timestamp_utc,
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-13",
                "split": "walk-forward",
                "parent_config_hash": parent_config_hash,
            },
            "status": status,
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score / 2.0,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0 if viable else -35.0,
                "trades_per_year": trades_per_year if trades_per_year is not None else (14.0 if viable else 1.0),
                "exposure": exposure if exposure is not None else (0.55 if viable else 0.05),
            },
            "robustness": {"negative_windows": 0 if viable else 4, "viable": viable},
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "baseline_verified": True,
                "baseline_metric_source": "verified_current_engine",
                "comparison_kind": "partial",
                "delta_sharpe": objective_score - 1.0,
                "delta_calmar": objective_score - 1.0,
                "delta_return": objective_score * 10.0,
                "beats_baseline_objective": objective_score >= 1.0,
                "beats_baseline_guardrails": objective_score >= 1.0,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class BranchDecayTests(unittest.TestCase):
    def test_failing_branch_is_retired_and_loses_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="active_root",
                family="momentum",
                config_hash="active_root",
                objective_score=1.0,
                viable=True,
                timestamp_utc="2026-04-01T00:00:00+00:00",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="active_child",
                family="momentum",
                config_hash="active_child",
                objective_score=1.3,
                viable=True,
                timestamp_utc="2026-04-10T00:00:00+00:00",
                parent_config_hash="active_root",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_root",
                family="momentum",
                config_hash="weak_root",
                objective_score=0.2,
                viable=False,
                timestamp_utc="2026-01-01T00:00:00+00:00",
                status="invalid",
                exposure=0.01,
                trades_per_year=0.0,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_child_1",
                family="momentum",
                config_hash="weak_child_1",
                objective_score=-0.2,
                viable=False,
                timestamp_utc="2026-01-10T00:00:00+00:00",
                status="no_trades",
                parent_config_hash="weak_root",
                exposure=0.0,
                trades_per_year=0.0,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="weak_child_2",
                family="momentum",
                config_hash="weak_child_2",
                objective_score=-0.3,
                viable=False,
                timestamp_utc="2026-01-15T00:00:00+00:00",
                status="invalid",
                parent_config_hash="weak_root",
                exposure=0.0,
                trades_per_year=0.0,
            )

            lineage_summary = build_lineage_summary(
                __import__("experiment_store").load_results_index(str(experiments_dir)),
                latest_batch={"summary": {"timestamp_utc": "2026-04-13T00:00:00+00:00"}},
            )
            branch_budget_plan, rationale = build_branch_budget_plan(
                lineage_summary,
                {"momentum": 10},
            )

        weak_branch = next(item for item in branch_budget_plan["momentum"] if item["branch_root_config_hash"] == "weak_root")
        active_branch = next(item for item in branch_budget_plan["momentum"] if item["branch_root_config_hash"] == "active_root")
        self.assertIn(weak_branch["branch_state"], {"paused", "retired"})
        self.assertEqual(weak_branch["branch_budget"], 0)
        self.assertGreater(active_branch["branch_budget"], weak_branch["branch_budget"])
        self.assertIn("branch_retirement_reason", weak_branch)
        self.assertIn("branch_revival_condition", weak_branch)
        self.assertIn("branch_decay_score", weak_branch)
        self.assertIn("branch_budget_rationale", rationale["families"]["momentum"]["branches"][0])

    def test_retired_branch_can_revive_with_new_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="root_old",
                family="momentum",
                config_hash="root_old",
                objective_score=0.2,
                viable=False,
                timestamp_utc="2026-01-01T00:00:00+00:00",
                status="invalid",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="root_new_child",
                family="momentum",
                config_hash="root_new_child",
                objective_score=1.4,
                viable=True,
                timestamp_utc="2026-04-12T00:00:00+00:00",
                parent_config_hash="root_old",
            )
            persisted = {
                "momentum": {
                    "root_old": {
                        "branch_state": "retired",
                        "branch_decay_score": 0.95,
                        "best_descendant_objective_score": 0.2,
                        "descendant_count": 1,
                        "confirmation_descendant_count": 0,
                        "holdout_descendant_count": 0,
                        "rejected_descendant_count": 1,
                        "history": [
                            {
                                "timestamp_utc": "2026-03-01T00:00:00+00:00",
                                "experiment_id": "root_old",
                                "branch_state": "retired",
                                "branch_decay_score": 0.95,
                            }
                        ],
                    }
                }
            }

            lineage_summary = build_lineage_summary(
                __import__("experiment_store").load_results_index(str(experiments_dir)),
                persisted_records=persisted,
                latest_batch={"summary": {"timestamp_utc": "2026-04-13T00:00:00+00:00"}},
            )
            branch_budget_plan, _ = build_branch_budget_plan(lineage_summary, {"momentum": 10})

        root_branch = next(item for item in branch_budget_plan["momentum"] if item["branch_root_config_hash"] == "root_old")
        self.assertEqual(root_branch["branch_state"], "revivable")
        self.assertGreater(root_branch["branch_budget"], 0)
        self.assertIn("branch_revival_condition", root_branch)

    def test_scorecards_and_dashboard_expose_branch_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="branch_seed",
                family="momentum",
                config_hash="branch_seed",
                objective_score=1.1,
                viable=True,
                timestamp_utc="2026-04-12T00:00:00+00:00",
            )
            scorecards = build_family_scorecards(families=["momentum"], base_dir=str(experiments_dir))
            dashboard = build_best_results_dashboard(base_dir=str(experiments_dir), families=["momentum"])

        scorecard = scorecards["momentum"]
        self.assertTrue(hasattr(scorecard, "branch_state"))
        self.assertTrue(hasattr(scorecard, "branch_decay_score"))
        self.assertTrue(hasattr(scorecard, "branch_retirement_reason"))
        self.assertTrue(hasattr(scorecard, "branch_revival_condition"))
        self.assertIn("branch_state", dashboard.top_overall[0])
        self.assertIn("branch_decay_score", dashboard.top_overall[0])


if __name__ == "__main__":
    unittest.main()
