import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline_registry import (
    get_baseline,
    load_baseline_registry,
    reproduce_baseline,
    save_verified_baseline_result,
)


class BaselineRegistryTests(unittest.TestCase):
    def test_registry_loads_s10005_baseline(self):
        records = load_baseline_registry()
        self.assertTrue(any(record["name"] == "momentum_champion_s10005" for record in records))

    def test_s10005_baseline_preserves_executable_vs_comment_distinction(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = get_baseline("momentum_champion_s10005", verified_dir=Path(tmp) / "verified")

        self.assertEqual(baseline["baseline_config_source"], "executable_code")
        self.assertEqual(baseline["config_source"], "executable_code")
        self.assertEqual(baseline["source_of_truth"], "uploaded reference file executable code")
        self.assertEqual(baseline["historical_provenance"]["historical_note_source"], "file_comment")
        self.assertEqual(baseline["historical_metrics"]["val"], 1.643)
        self.assertIn("provenance only", baseline["historical_metrics_note"])
        self.assertIsNone(baseline["metrics_verified_by_current_engine"])
        self.assertEqual(baseline["config"]["FG_MIN"], 10.0)
        self.assertEqual(baseline["config"]["STOP_PARABOLIC"], 0.3)
        self.assertEqual(baseline["config"]["STOP_LOSS_PCT"], 0.2)
        self.assertEqual(baseline["config"]["EXIT_PCT_RANK"], 0.97)
        self.assertEqual(baseline["historical_provenance"]["starting_point_champion_note"]["reported_val"], 1.643)
        self.assertFalse(baseline["comparison_policy"]["comparison_eligible"])

    def test_verified_metrics_are_loaded_separately_from_historical_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            verified_dir = Path(tmp) / "verified"
            save_verified_baseline_result(
                "momentum_champion_s10005",
                {
                    "spec": {
                        "timestamp_utc": "2026-04-03T00:00:00+00:00",
                        "experiment_id": "exp1",
                        "strategy_family": "momentum",
                        "config_hash": "abc123",
                        "dataset_id": "data1",
                        "data_start": "2020-01-01",
                        "data_end": "2020-12-31",
                        "split": "walk-forward",
                        "benchmark_source": "spy_symbol",
                        "git_commit": "deadbeef",
                        "family_version": "test",
                    },
                    "status": "success",
                    "objective_score": 1.23,
                    "metrics": {"sharpe": 0.9, "total_return": 12.5},
                    "robustness": {"viable": True},
                    "runtime_seconds": 0.1,
                    "artifacts": {},
                },
                baselines_dir=verified_dir,
            )
            baseline = get_baseline("momentum_champion_s10005", verified_dir=verified_dir)

        self.assertEqual(baseline["verified_metrics"]["sharpe"], 0.9)
        self.assertEqual(
            baseline["historical_provenance"]["starting_point_champion_note"]["verification_status"],
            "not_verified_as_s10005_runnable_config",
        )
        self.assertEqual(baseline["metrics_verified_by_current_engine"]["sharpe"], 0.9)
        self.assertEqual(baseline["comparison_policy"]["metrics_verification_status"], "verified_current_engine")
        self.assertTrue(baseline["comparison_policy"]["comparison_eligible"])

    def test_reproduction_returns_current_engine_baseline_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            verified_dir = Path(tmp) / "verified"
            fake_result = type(
                "FakeResult",
                (),
                {
                    "status": "success",
                    "objective_score": 1.1,
                    "metrics": {"sharpe": 0.8, "total_return": 10.0},
                    "robustness": {"viable": True},
                    "artifacts": {},
                    "runtime_seconds": 0.2,
                    "spec": type(
                        "FakeSpec",
                        (),
                        {
                            "timestamp_utc": "2026-04-03T00:00:00+00:00",
                            "experiment_id": "exp2",
                            "strategy_family": "momentum",
                            "config_hash": "cfg1",
                            "dataset_id": "data1",
                            "data_start": "2020-01-01",
                            "data_end": "2020-12-31",
                            "split": "walk-forward",
                            "benchmark_source": "spy_symbol",
                            "git_commit": "deadbeef",
                            "family_version": "test",
                        },
                    )(),
                },
            )()
            with patch("experiment_runner.run_single_experiment", return_value=fake_result):
                payload = reproduce_baseline(
                    "momentum_champion_s10005",
                    data={},
                    persist=False,
                    verified_dir=verified_dir,
                )

            self.assertEqual(payload["result"].status, "success")
            self.assertEqual(payload["baseline"]["verified_metrics"]["sharpe"], 0.8)


if __name__ == "__main__":
    unittest.main()
