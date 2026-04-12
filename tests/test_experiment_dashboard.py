import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_dashboard import build_best_results_dashboard, format_dashboard_markdown, save_dashboard_reports
from experiment_store import init_store, save_experiment_result


def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    objective_score: float,
    viable: bool,
    status: str = "success",
    timestamp_utc: str = "2026-04-12T00:00:00+00:00",
    beats_baseline: bool | None = None,
) -> None:
    baseline = None
    if beats_baseline is not None:
        baseline = {
            "baseline_name": "momentum_champion_s10005",
            "comparison_status": "partial_verified_current_engine",
            "baseline_verified": True,
            "baseline_metric_source": "current_engine",
            "comparison_kind": "partial",
            "delta_sharpe": objective_score - 1.0,
            "delta_calmar": objective_score - 1.0,
            "delta_return": objective_score * 10.0,
            "beats_baseline_objective": beats_baseline,
            "beats_baseline_guardrails": beats_baseline,
        }
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"X": objective_score},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "dashboard_test",
                "config_hash": f"{experiment_id}_hash",
                "experiment_id": experiment_id,
                "timestamp_utc": timestamp_utc,
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-12",
                "split": "walk-forward",
                "strategy_type": "classical",
                "source_proposal_id": "proposal_dashboard",
                "source_idea_ids": ["idea_dashboard"],
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
            "baseline_comparison": baseline,
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


class ExperimentDashboardTests(unittest.TestCase):
    def test_dashboard_ranking_views_and_scorecards(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(
                tmp,
                experiment_id="momentum_best",
                family="momentum",
                objective_score=2.0,
                viable=True,
                beats_baseline=True,
                timestamp_utc="2026-04-12T00:00:01+00:00",
            )
            _save_result(
                tmp,
                experiment_id="momentum_weaker",
                family="momentum",
                objective_score=1.0,
                viable=True,
                beats_baseline=False,
                timestamp_utc="2026-04-12T00:00:02+00:00",
            )
            _save_result(
                tmp,
                experiment_id="rl_non_viable",
                family="rl_bandit",
                objective_score=9.0,
                viable=False,
                beats_baseline=False,
                timestamp_utc="2026-04-12T00:00:03+00:00",
            )
            _save_result(
                tmp,
                experiment_id="duplicate_skip",
                family="momentum",
                objective_score=99.0,
                viable=True,
                status="duplicate",
                timestamp_utc="2026-04-12T00:00:04+00:00",
            )

            dashboard = build_best_results_dashboard(base_dir=tmp, overall_limit=20, viable_limit=20, per_family_limit=10)

        self.assertEqual(dashboard.top_overall[0]["experiment_id"], "momentum_best")
        self.assertNotIn("duplicate_skip", {row["experiment_id"] for row in dashboard.top_overall})
        self.assertEqual([row["experiment_id"] for row in dashboard.top_viable], ["momentum_best", "momentum_weaker"])
        self.assertEqual([row["experiment_id"] for row in dashboard.top_baseline_beating], ["momentum_best"])
        self.assertEqual(dashboard.top_per_family["momentum"][0]["experiment_id"], "momentum_best")
        self.assertIn("momentum", dashboard.family_scorecards)
        self.assertEqual(dashboard.family_scorecards["momentum"]["total_experiments"], 3)

    def test_dashboard_latest_non_empty_batch_excludes_empty_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            old_full = Path(tmp) / "batches" / "batch_20260411_full"
            old_full.mkdir(parents=True, exist_ok=True)
            (old_full / "summary.json").write_text(json.dumps({"total_executed": 2}))
            pd.DataFrame([{"experiment_id": "x", "status": "success"}]).to_csv(old_full / "leaderboard.csv", index=False)

            latest_empty = Path(tmp) / "batches" / "batch_20260412_empty"
            latest_empty.mkdir(parents=True, exist_ok=True)
            (latest_empty / "summary.json").write_text(json.dumps({"total_executed": 0}))
            pd.DataFrame(columns=["experiment_id", "status"]).to_csv(latest_empty / "leaderboard.csv", index=False)

            dashboard = build_best_results_dashboard(base_dir=tmp)

        self.assertIsNotNone(dashboard.latest_non_empty_batch)
        self.assertEqual(dashboard.latest_non_empty_batch["batch_id"], "batch_20260411_full")

    def test_dashboard_persistence_writes_json_markdown_and_scorecards(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiments_dir = Path(tmp) / "experiments"
            reports_dir = Path(tmp) / "reports"
            init_store(str(experiments_dir))
            _save_result(
                str(experiments_dir),
                experiment_id="m1",
                family="momentum",
                objective_score=1.5,
                viable=True,
                beats_baseline=True,
            )

            dashboard = build_best_results_dashboard(base_dir=str(experiments_dir))
            paths = save_dashboard_reports(dashboard, reports_dir=str(reports_dir))

            for path in paths.values():
                self.assertTrue(Path(path).exists())
            payload = json.loads(Path(paths["best_results_json"]).read_text())
            self.assertIn("top_overall", payload)
            self.assertIn("family_scorecards", payload)
            self.assertIn("momentum", json.loads(Path(paths["family_scorecards_json"]).read_text())["families"])
            self.assertIn("Top 1 Overall", Path(paths["best_results_md"]).read_text())

    def test_dashboard_gracefully_handles_missing_historical_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(tmp, experiment_id="old", family="momentum", objective_score=1.0, viable=True)
            index_path = Path(tmp) / "index.csv"
            index = pd.read_csv(index_path)
            index = index.drop(columns=["strategy_type", "baseline_name", "beats_baseline_objective"])
            index.to_csv(index_path, index=False)

            dashboard = build_best_results_dashboard(base_dir=tmp)
            markdown = format_dashboard_markdown(dashboard)

        self.assertEqual(dashboard.top_overall[0]["experiment_id"], "old")
        self.assertEqual(dashboard.top_baseline_beating, [])
        self.assertIn("old", markdown)


if __name__ == "__main__":
    unittest.main()
