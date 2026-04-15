import json
import tempfile
import unittest
from pathlib import Path

from experiment_idea_yield import build_idea_yield_summary, load_latest_idea_yield_summary, save_idea_yield_summary
from experiment_store import init_store, save_experiment_result


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    objective_score: float,
    viable: bool,
    source_type: str = "template_expansion",
    idea_kind: str = "template_variation",
    template_id: str = "template_a",
    beats_baseline: bool = False,
    status: str = "success",
    timestamp_utc: str = "2026-04-12T00:00:00+00:00",
) -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "strategy_type": "classical",
                "source_type": source_type,
                "idea_kind": idea_kind,
                "template_id": template_id,
                "params": {"X": objective_score},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "idea_yield_test",
                "config_hash": f"{experiment_id}_hash",
                "experiment_id": experiment_id,
                "timestamp_utc": timestamp_utc,
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-12",
                "split": "walk-forward",
            },
            "status": status,
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score / 2.0,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
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


class IdeaYieldTests(unittest.TestCase):
    def test_idea_yield_summary_is_built_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="m1",
                family="momentum",
                objective_score=1.4,
                viable=True,
                beats_baseline=True,
                source_type="template_expansion",
                idea_kind="template_variation",
                template_id="mom_template_a",
                timestamp_utc="2026-04-10T00:00:00+00:00",
            )
            _save_result(
                str(experiments_dir),
                experiment_id="m2",
                family="momentum",
                objective_score=-0.2,
                viable=False,
                source_type="history_mining",
                idea_kind="branch_refinement",
                template_id="mom_template_b",
                timestamp_utc="2026-04-11T00:00:00+00:00",
            )

            summary = build_idea_yield_summary(families=["momentum"], base_dir=str(experiments_dir), persist_memory=True)
            path = save_idea_yield_summary(summary, base_dir=str(experiments_dir))
            self.assertTrue(path.exists())
            loaded = load_latest_idea_yield_summary(str(experiments_dir))

        self.assertIsNotNone(loaded)
        family = summary["families"]["momentum"]
        self.assertEqual(family["idea_attempt_count"], 2)
        self.assertIn(family["idea_state"], {"active", "promising", "stale", "retired", "untested"})
        self.assertGreaterEqual(family["idea_quality_score"], 0.0)
        self.assertIn("by_idea_kind", summary)
        self.assertIn("by_template_id", summary)
        self.assertIn("by_strategy_type", summary)
        self.assertIn("by_idea_source", summary)
        self.assertEqual(loaded["families"]["momentum"]["idea_attempt_count"], 2)


if __name__ == "__main__":
    unittest.main()
