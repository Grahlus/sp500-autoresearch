import tempfile
import unittest
from pathlib import Path

from experiment_store import (
    compute_config_hash,
    has_experiment_result,
    init_store,
    load_experiment_result,
    load_results_index,
    save_experiment_result,
)


class ExperimentStoreTests(unittest.TestCase):
    def test_hash_is_stable_for_equivalent_configs(self):
        a = compute_config_hash("momentum", {"A": 1, "B": 2})
        b = compute_config_hash("momentum", {"B": 2, "A": 1})
        self.assertEqual(a, b)

    def test_persistence_writes_expected_files_and_updates_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_store(tmp)
            result = {
                "spec": {
                    "family": "momentum",
                    "params": {"LOOKBACK_WEEKS": 26},
                    "search_method": "single",
                    "objective_name": "wf_v1_score",
                    "batch_id": "single",
                    "config_hash": "abc123",
                    "experiment_id": "momentum_abc123",
                    "timestamp_utc": "2026-04-03T00:00:00+00:00",
                    "benchmark_source": "spy_symbol",
                    "dataset_id": "data123",
                    "data_start": "2014-01-01",
                    "data_end": "2026-04-02",
                    "split": "walk-forward",
                    "git_commit": "deadbee",
                    "family_version": "v1",
                },
                "status": "success",
                "objective_score": 1.0,
                "metrics": {"sharpe": 1.0, "total_return": 10.0},
                "robustness": {"negative_windows": 0, "viable": True},
                "artifacts": {},
                "error_message": None,
                "runtime_seconds": 0.1,
            }
            save_experiment_result(result, base_dir=tmp)

            self.assertTrue((Path(tmp) / "runs" / "momentum_abc123" / "spec.json").exists())
            self.assertTrue((Path(tmp) / "runs" / "momentum_abc123" / "metrics.json").exists())
            self.assertTrue((Path(tmp) / "runs" / "momentum_abc123" / "result.json").exists())
            self.assertTrue(has_experiment_result("abc123", family="momentum", base_dir=tmp))

            index = load_results_index(tmp)
            self.assertEqual(len(index), 1)
            self.assertEqual(index.iloc[0]["strategy_family"], "momentum")

            loaded = load_experiment_result("abc123", family="momentum", base_dir=tmp)
            self.assertEqual(loaded["status"], "success")
            self.assertEqual(loaded["spec"]["strategy_family"], "momentum")
            self.assertEqual(loaded["spec"]["config"]["LOOKBACK_WEEKS"], 26)


if __name__ == "__main__":
    unittest.main()
