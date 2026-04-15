import tempfile
import unittest
from pathlib import Path

from experiment_dashboard import build_best_results_dashboard
from experiment_lineage import build_lineage_summary
from experiment_memory import load_research_memory
from experiment_runtime_decision import build_runtime_decision, save_runtime_decision
from experiment_store import init_store, load_results_index, save_experiment_result
from experiment_types import RuntimeDecisionInput


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    config_hash: str,
    objective_score: float,
    viable: bool = True,
    status: str = "success",
    parent_config_hash: str | None = None,
    confirmation_required: bool = False,
    confirmation_state: str | None = None,
    confirmation_outcome: str | None = None,
    holdout_check_required: bool = False,
    holdout_check_type: str | None = None,
    holdout_check_outcome: str | None = None,
    holdout_check_scope: str | None = None,
) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"X": objective_score},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "lineage_batch",
                "config_hash": config_hash,
                "experiment_id": experiment_id,
                "timestamp_utc": "2026-04-13T00:00:00+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-13",
                "split": "walk-forward",
                "parent_config_hash": parent_config_hash,
                "confirmation_required": confirmation_required,
                "confirmation_state": confirmation_state,
                "confirmation_outcome": confirmation_outcome,
                "holdout_check_required": holdout_check_required,
                "holdout_check_type": holdout_check_type,
                "holdout_check_outcome": holdout_check_outcome,
                "holdout_check_scope": holdout_check_scope,
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
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "beats_baseline_objective": objective_score >= 1.0,
                "beats_baseline_guardrails": objective_score >= 1.0,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class LineageTests(unittest.TestCase):
    def test_lineage_summary_tracks_parent_child_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="root",
                family="momentum",
                config_hash="root_hash",
                objective_score=1.0,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="confirm",
                family="momentum",
                config_hash="confirm_hash",
                objective_score=1.2,
                parent_config_hash="root_hash",
                confirmation_required=True,
                confirmation_state="confirmed",
                confirmation_outcome="confirmed",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="holdout",
                family="momentum",
                config_hash="holdout_hash",
                objective_score=1.1,
                parent_config_hash="root_hash",
                holdout_check_required=True,
                holdout_check_type="long_horizon_holdout",
                holdout_check_outcome="confirmed",
                holdout_check_scope="long_horizon",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="reject",
                family="momentum",
                config_hash="reject_hash",
                objective_score=-0.2,
                viable=False,
                status="invalid",
                parent_config_hash="root_hash",
            )

            lineage = build_lineage_summary(load_results_index(str(experiments_dir)))
            dashboard = build_best_results_dashboard(base_dir=str(experiments_dir), overall_limit=10, viable_limit=10, per_family_limit=5)

        root_record = lineage["by_config_hash"]["root_hash"]
        self.assertEqual(root_record["descendant_count"], 3)
        self.assertEqual(root_record["confirmation_descendant_count"], 1)
        self.assertEqual(root_record["holdout_descendant_count"], 1)
        self.assertEqual(root_record["rejected_descendant_count"], 1)
        self.assertIn(root_record["lineage_status_summary"], {"strengthening", "mixed", "failing"})
        self.assertIn("lineage_root_config_hash", dashboard.top_overall[0])
        self.assertIn("lineage_status_summary", dashboard.top_overall[0])
        self.assertIn("lineage_trust_score", dashboard.family_scorecards["momentum"])
        self.assertIn("lineage_status_summary", dashboard.family_scorecards["momentum"])

    def test_runtime_decision_persists_lineage_state_across_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="root",
                family="momentum",
                config_hash="root_hash",
                objective_score=1.0,
            )
            _save_result(
                str(experiments_dir),
                experiment_id="child",
                family="momentum",
                config_hash="child_hash",
                objective_score=1.3,
                parent_config_hash="root_hash",
                confirmation_required=True,
                confirmation_state="confirmed",
                confirmation_outcome="confirmed",
            )

            decision = build_runtime_decision(
                RuntimeDecisionInput(
                    workspace_root=tmp,
                    experiments_dir=str(experiments_dir),
                    strategy_families=["momentum"],
                    max_experiments=8,
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                )
            )
            save_runtime_decision(decision, workspace_root=tmp)
            memory = load_research_memory(str(experiments_dir))

        lineage_states = memory.get("lineage_states") or {}
        self.assertIn("momentum", lineage_states)
        self.assertIn("root_hash", lineage_states["momentum"])
        self.assertGreaterEqual(int(lineage_states["momentum"]["root_hash"]["descendant_count"]), 1)
        self.assertIn("lineage_summary", decision.rationale)
        self.assertIn("family_summaries", decision.rationale["lineage_summary"])


if __name__ == "__main__":
    unittest.main()
