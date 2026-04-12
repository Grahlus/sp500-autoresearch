import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiment_best_results import (
    ensure_global_results_index,
    latest_non_empty_batch,
    save_best_results_reports,
    top_results_overall,
    top_results_per_family,
)
from experiment_store import init_store, save_experiment_result


def _save_result(tmp: str, experiment_id: str, family: str, config_hash: str, objective_score: float, status: str = "success") -> None:
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": {"P": objective_score},
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "single",
                "config_hash": config_hash,
                "experiment_id": experiment_id,
                "timestamp_utc": f"2026-04-04T00:00:0{len(experiment_id)}+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2014-01-01",
                "data_end": "2026-04-03",
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
            "robustness": {"negative_windows": 0, "viable": True},
            "artifacts": {},
            "error_message": None,
            "runtime_seconds": 0.1,
        },
        base_dir=tmp,
    )


class BestResultsTests(unittest.TestCase):
    def test_ensure_global_index_creates_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = ensure_global_results_index(tmp)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "index.csv")

    def test_top_results_are_sorted_and_grouped(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(tmp, "m2", "momentum", "m2", 2.0)
            _save_result(tmp, "m1", "momentum", "m1", 1.0)
            _save_result(tmp, "s1", "superstock", "s1", 1.5)
            _save_result(tmp, "d1", "momentum", "d1", 99.0, status="duplicate")

            overall = top_results_overall(base_dir=tmp)
            self.assertEqual(list(overall["experiment_id"][:3]), ["m2", "s1", "m1"])
            self.assertNotIn("d1", set(overall["experiment_id"]))

            per_family = top_results_per_family(base_dir=tmp)
            self.assertEqual(list(per_family["momentum"]["experiment_id"]), ["m2", "m1"])
            self.assertEqual(list(per_family["superstock"]["experiment_id"]), ["s1"])

    def test_best_results_rank_viable_before_non_viable_even_with_higher_objective(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(tmp, "weak", "rl_bandit", "weak", 2.5)
            _save_result(tmp, "robust", "momentum", "robust", 1.5)

            index_path = Path(tmp) / "index.csv"
            index = pd.read_csv(index_path)
            index.loc[index["experiment_id"] == "weak", "viable"] = False
            index.loc[index["experiment_id"] == "robust", "viable"] = True
            index.to_csv(index_path, index=False)

            overall = top_results_overall(base_dir=tmp)
            self.assertEqual(overall.iloc[0]["experiment_id"], "robust")
            self.assertEqual(overall.iloc[1]["experiment_id"], "weak")

    def test_missing_optional_metadata_is_displayed_as_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(tmp, "old", "momentum", "old", 1.0)

            overall = top_results_overall(base_dir=tmp)
            for column in ("strategy_type", "source_type", "template_id", "hypothesis", "reason_selected"):
                self.assertEqual(overall.iloc[0][column], "")

    def test_latest_non_empty_batch_skips_empty_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            empty_dir = Path(tmp) / "batches" / "batch_empty"
            empty_dir.mkdir(parents=True, exist_ok=True)
            (empty_dir / "summary.json").write_text(json.dumps({"total_executed": 0}))
            pd.DataFrame(columns=["experiment_id", "status"]).to_csv(empty_dir / "leaderboard.csv", index=False)

            full_dir = Path(tmp) / "batches" / "batch_full"
            full_dir.mkdir(parents=True, exist_ok=True)
            (full_dir / "summary.json").write_text(json.dumps({"total_executed": 2}))
            pd.DataFrame([{"experiment_id": "x", "status": "success"}]).to_csv(full_dir / "leaderboard.csv", index=False)

            latest = latest_non_empty_batch(base_dir=tmp)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["batch_id"], "batch_full")
            self.assertEqual(latest["executed_count"], 2)

    def test_save_best_results_reports_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            _save_result(tmp, "m1", "momentum", "m1", 1.0)
            paths = save_best_results_reports(base_dir=tmp, overall_limit=20, per_family_limit=10)
            self.assertTrue(Path(paths["overall_path"]).exists())
            self.assertTrue(Path(paths["per_family_path"]).exists())
            self.assertTrue(Path(paths["latest_batch_path"]).exists())


if __name__ == "__main__":
    unittest.main()
