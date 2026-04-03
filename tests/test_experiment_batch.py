import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from experiment_batch import (
    build_batch_leaderboard,
    build_batch_request,
    build_batch_summary,
    run_batch_experiments,
    sample_batch_configs,
)
from experiment_types import ExperimentResult, ExperimentSpec


class ExperimentBatchTests(unittest.TestCase):
    def test_batch_request_validates_cleanly(self):
        request = build_batch_request(strategy_families=["momentum"], sampler_type="random", max_experiments=3)
        self.assertEqual(request.strategy_families, ["momentum"])
        self.assertEqual(request.max_per_family, 3)

    def test_sampler_is_deterministic(self):
        a = sample_batch_configs("momentum", "random", 5, 7)
        b = sample_batch_configs("momentum", "random", 5, 7)
        self.assertEqual(a, b)

    def test_sampler_produces_family_valid_configs(self):
        configs = sample_batch_configs("superstock", "random", 3, 11)
        self.assertEqual(len(configs), 3)
        for config in configs:
            self.assertIn("max_positions", config)
            self.assertLessEqual(config["price_min"], config["price_max"])

    def test_batch_runner_calls_single_experiment_runner(self):
        request = build_batch_request(strategy_families=["momentum"], sampler_type="random", max_experiments=2, seed=1)
        fake_result = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="abc", experiment_id="e1"),
            status="success",
            objective_score=1.0,
            metrics={"sharpe": 1.0},
            robustness={"viable": True},
            artifacts={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiment_batch.load_data", return_value={}), patch(
                "experiment_batch.run_single_experiment", return_value=fake_result
            ) as mock_runner:
                result = run_batch_experiments(request, base_dir=tmp)
        self.assertEqual(mock_runner.call_count, 2)
        self.assertEqual(result.total_executed, 2)

    def test_resume_mode_skips_prior_successes(self):
        request = build_batch_request(strategy_families=["momentum"], sampler_type="random", max_experiments=2, seed=1, resume=True)
        prior = pd.DataFrame([{"config_hash": "h1", "status": "success"}])
        fake_result = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="h2", experiment_id="e2"),
            status="success",
            objective_score=1.0,
            metrics={"sharpe": 1.0},
            robustness={"viable": True},
            artifacts={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiment_batch.load_data", return_value={}), patch(
                "experiment_batch.sample_batch_configs",
                return_value=[{"LOOKBACK_WEEKS": 20}, {"LOOKBACK_WEEKS": 26}],
            ), patch("experiment_batch.compute_config_hash", side_effect=["h1", "h2"]), patch(
                "experiment_batch.load_prior_results", return_value=prior
            ), patch("experiment_batch.run_single_experiment", return_value=fake_result) as mock_runner:
                result = run_batch_experiments(request, base_dir=tmp)
        self.assertEqual(mock_runner.call_count, 1)
        self.assertEqual(result.total_skipped, 1)
        self.assertEqual(result.total_executed, 1)

    def test_ranking_output_sorts_by_objective(self):
        r1 = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="a", experiment_id="a"),
            status="success",
            objective_score=1.5,
            metrics={"sharpe": 1.0, "calmar": 1.0, "total_return": 10.0},
            robustness={"viable": True},
            artifacts={},
        )
        r2 = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="b", experiment_id="b"),
            status="success",
            objective_score=0.5,
            metrics={"sharpe": 0.4, "calmar": 0.5, "total_return": 5.0},
            robustness={"viable": True},
            artifacts={},
        )
        leaderboard = build_batch_leaderboard([r2, r1])
        self.assertEqual(leaderboard.iloc[0]["experiment_id"], "a")

    def test_baseline_comparison_fields_appear_in_ranked_results(self):
        result = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="a", experiment_id="a"),
            status="success",
            objective_score=1.0,
            metrics={"sharpe": 1.0, "calmar": 1.0, "total_return": 10.0},
            robustness={"viable": True},
            artifacts={},
            baseline_comparison={
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "exact_verified_current_engine",
                "delta_sharpe": 0.1,
                "delta_calmar": 0.2,
                "delta_return": 1.0,
                "beats_baseline_objective": True,
                "beats_baseline_guardrails": True,
            },
        )
        leaderboard = build_batch_leaderboard([result])
        self.assertEqual(leaderboard.iloc[0]["baseline_name"], "momentum_champion_s10005")
        self.assertEqual(leaderboard.iloc[0]["comparison_status"], "exact_verified_current_engine")

    def test_batch_summary_includes_per_family_counts(self):
        request = build_batch_request(strategy_families=["momentum"], max_experiments=1)
        result = ExperimentResult(
            spec=ExperimentSpec(family="momentum", params={}, config_hash="a", experiment_id="a"),
            status="success",
            objective_score=1.0,
            metrics={"sharpe": 1.0},
            robustness={"viable": True},
            artifacts={},
        )
        summary = build_batch_summary(
            type(
                "FakeBatch",
                (),
                {
                    "request": request,
                    "total_sampled": 1,
                    "total_executed": 1,
                    "total_skipped": 0,
                    "total_failed": 0,
                    "results": [result],
                },
            )()
        )
        self.assertEqual(summary["family_summary"]["momentum"]["executed"], 1)


if __name__ == "__main__":
    unittest.main()
