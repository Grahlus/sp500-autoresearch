import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiment_runner import build_strategy_weights_for_experiment, run_single_experiment
from baseline_registry import save_verified_baseline_result
from strategies.base import StrategyFamily


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2024-01-01", periods=40)
        close = pd.DataFrame({"AAA": [10.0] * len(idx), "SPY": [100.0] * len(idx)}, index=idx)
        self.data = {
            "open": close.copy(),
            "high": close.copy(),
            "low": close.copy(),
            "close": close.copy(),
            "volume": close.copy() * 1000,
            "vix": pd.Series([20.0] * len(idx), index=idx),
            "fear_greed": pd.Series([50.0] * len(idx), index=idx),
        }

    def test_build_strategy_weights_prefers_config_hook(self):
        family = StrategyFamily(
            name="fake",
            metric="sharpe",
            hypothesis="test",
            generate_signals=lambda data: data["close"] * 0.0,
            generate_signals_with_config=lambda data, config: pd.DataFrame(
                1.0, index=data["close"].index, columns=data["close"].columns
            ),
        )
        with patch("experiment_runner.get_strategy_family", return_value=family):
            weights = build_strategy_weights_for_experiment("fake", self.data, {"x": 1})
        self.assertEqual(float(weights.iloc[0, 0]), 1.0)

    def test_single_experiment_uses_backtest_and_returns_structured_result(self):
        family = StrategyFamily(
            name="fake",
            metric="sharpe",
            hypothesis="test-version",
            generate_signals=lambda data: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
            generate_signals_with_config=lambda data, config: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
        )
        fake_metrics = {
            "sharpe": 0.5,
            "calmar": 0.5,
            "total_return_pct": 5.0,
            "trades_per_year": 10.0,
            "max_drawdown": -5.0,
            "sharpe_min": 0.1,
            "windows": [{"sharpe": 0.5, "final_value": 101_000, "n_days": 126}],
            "windows_beat_spy": "1/1",
        }
        with patch("experiment_runner.get_strategy_family", return_value=family), patch(
            "experiment_runner.normalize_experiment_config", return_value={}
        ), patch("experiment_runner.validate_experiment_config", return_value=(True, None)), patch(
            "experiment_runner.run_backtest", return_value=fake_metrics
        ):
            result = run_single_experiment("fake", {}, data=self.data, persist=False)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.spec.strategy_family, "fake")
        self.assertEqual(result.metrics["total_return"], 5.0)
        self.assertIsNotNone(result.objective_score)
        self.assertEqual(result.spec.benchmark_source, "spy_symbol")

    def test_invalid_unknown_key_fails_cleanly(self):
        result = run_single_experiment("momentum", {"NOT_A_PARAM": 1}, data=self.data, persist=False)
        self.assertEqual(result.status, "invalid")
        self.assertIn("Unknown config key", result.error_message)

    def test_persistence_updates_index(self):
        family = StrategyFamily(
            name="fake",
            metric="sharpe",
            hypothesis="test-version",
            generate_signals=lambda data: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
            generate_signals_with_config=lambda data, config: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
        )
        fake_metrics = {
            "sharpe": 0.5,
            "calmar": 0.5,
            "total_return_pct": 5.0,
            "trades_per_year": 10.0,
            "max_drawdown": -5.0,
            "sharpe_min": 0.1,
            "windows": [{"sharpe": 0.5, "final_value": 101_000, "n_days": 126}],
            "windows_beat_spy": "1/1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiment_runner.get_strategy_family", return_value=family), patch(
                "experiment_runner.normalize_experiment_config", return_value={}
            ), patch("experiment_runner.validate_experiment_config", return_value=(True, None)), patch(
                "experiment_runner.run_backtest", return_value=fake_metrics
            ):
                result = run_single_experiment("fake", {}, data=self.data, persist=True, base_dir=tmp)
            self.assertEqual(result.status, "success")
            self.assertTrue((pd.read_csv(f"{tmp}/index.csv")["strategy_family"] == "fake").any())

    def test_duplicate_successful_config_is_not_rerun(self):
        family = StrategyFamily(
            name="fake",
            metric="sharpe",
            hypothesis="test-version",
            generate_signals=lambda data: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
            generate_signals_with_config=lambda data, config: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
        )
        fake_metrics = {
            "sharpe": 0.5,
            "calmar": 0.5,
            "total_return_pct": 5.0,
            "trades_per_year": 10.0,
            "max_drawdown": -5.0,
            "sharpe_min": 0.1,
            "windows": [{"sharpe": 0.5, "final_value": 101_000, "n_days": 126}],
            "windows_beat_spy": "1/1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("experiment_runner.get_strategy_family", return_value=family), patch(
                "experiment_runner.normalize_experiment_config", return_value={}
            ), patch("experiment_runner.validate_experiment_config", return_value=(True, None)), patch(
                "experiment_runner.run_backtest", return_value=fake_metrics
            ) as mock_backtest:
                first = run_single_experiment("fake", {}, data=self.data, persist=True, base_dir=tmp)
                second = run_single_experiment("fake", {}, data=self.data, persist=True, base_dir=tmp)
            self.assertEqual(first.status, "success")
            self.assertEqual(second.status, "duplicate")
            self.assertEqual(mock_backtest.call_count, 1)

    def test_no_trade_status_is_supported(self):
        family = StrategyFamily(
            name="fake",
            metric="sharpe",
            hypothesis="test-version",
            generate_signals=lambda data: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
            generate_signals_with_config=lambda data, config: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
        )
        fake_metrics = {
            "sharpe": 0.0,
            "calmar": 0.0,
            "total_return_pct": 0.0,
            "trades_per_year": 0.0,
            "max_drawdown": 0.0,
            "sharpe_min": 0.0,
            "windows": [{"sharpe": 0.0, "final_value": 100_000, "n_days": 126}],
            "windows_beat_spy": "0/1",
        }
        with patch("experiment_runner.get_strategy_family", return_value=family), patch(
            "experiment_runner.normalize_experiment_config", return_value={}
        ), patch("experiment_runner.validate_experiment_config", return_value=(True, None)), patch(
            "experiment_runner.run_backtest", return_value=fake_metrics
        ):
            result = run_single_experiment("fake", {}, data=self.data, persist=False)
        self.assertEqual(result.status, "no_trades")

    def test_comparison_fields_distinguish_verified_vs_historical_baselines(self):
        family = StrategyFamily(
            name="momentum",
            metric="sharpe",
            hypothesis="test-version",
            generate_signals=lambda data: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
            generate_signals_with_config=lambda data, config: pd.DataFrame(0.0, index=data["close"].index, columns=data["close"].columns),
        )
        fake_metrics = {
            "sharpe": 0.5,
            "calmar": 0.5,
            "total_return_pct": 5.0,
            "trades_per_year": 10.0,
            "max_drawdown": -5.0,
            "sharpe_min": 0.1,
            "windows": [{"sharpe": 0.5, "final_value": 101_000, "n_days": 126}],
            "windows_beat_spy": "1/1",
        }
        with tempfile.TemporaryDirectory() as tmp:
            missing_verified_dir = Path(tmp) / "verified"
            with patch("experiment_runner.get_baseline") as mock_get_baseline, patch(
                "experiment_runner.get_strategy_family", return_value=family
            ), patch("experiment_runner.run_backtest", return_value=fake_metrics):
                from baseline_registry import get_baseline

                mock_get_baseline.side_effect = lambda name: get_baseline(name, verified_dir=missing_verified_dir)
                historical_only = run_single_experiment(
                    "momentum",
                    {},
                    data=self.data,
                    persist=False,
                    compare_to_baseline="momentum_champion_s10005",
                )
        self.assertEqual(historical_only.baseline_comparison["baseline_metric_source"], "historical_note_only")
        self.assertEqual(historical_only.baseline_comparison["comparison_kind"], "partial")
        self.assertEqual(historical_only.baseline_comparison["comparison_status"], "historical_provenance_only")
        self.assertFalse(historical_only.baseline_comparison["baseline_verified"])
        self.assertFalse(historical_only.baseline_comparison["comparison_eligible"])
        self.assertIsNone(historical_only.baseline_comparison["delta_sharpe"])

        with tempfile.TemporaryDirectory() as tmp:
            verified_dir = Path(tmp) / "verified"
            save_verified_baseline_result(
                "momentum_champion_s10005",
                {
                    "spec": {
                        "timestamp_utc": "2026-04-03T00:00:00+00:00",
                        "experiment_id": "exp1",
                        "strategy_family": "momentum",
                        "config_hash": historical_only.spec.config_hash,
                        "dataset_id": historical_only.spec.dataset_id,
                        "data_start": historical_only.spec.data_start,
                        "data_end": historical_only.spec.data_end,
                        "split": historical_only.spec.split,
                        "benchmark_source": historical_only.spec.benchmark_source,
                        "git_commit": "deadbeef",
                        "family_version": "test",
                    },
                    "status": "success",
                    "objective_score": 1.23,
                    "metrics": {"sharpe": 0.4, "calmar": 0.3, "total_return": 4.0},
                    "robustness": {"viable": True},
                    "runtime_seconds": 0.1,
                    "artifacts": {},
                },
                baselines_dir=verified_dir,
            )
            with patch("experiment_runner.get_baseline") as mock_get_baseline, patch(
                "experiment_runner.get_strategy_family", return_value=family
            ), patch("experiment_runner.run_backtest", return_value=fake_metrics):
                from baseline_registry import get_baseline

                mock_get_baseline.side_effect = lambda name: get_baseline(name, verified_dir=verified_dir)
                verified = run_single_experiment(
                    "momentum",
                    {},
                    data=self.data,
                    persist=False,
                    compare_to_baseline="momentum_champion_s10005",
                )

        self.assertEqual(verified.baseline_comparison["baseline_metric_source"], "verified_current_engine")
        self.assertEqual(verified.baseline_comparison["comparison_kind"], "exact")
        self.assertEqual(verified.baseline_comparison["comparison_status"], "exact_verified_current_engine")
        self.assertTrue(verified.baseline_comparison["baseline_verified"])
        self.assertTrue(verified.baseline_comparison["comparison_eligible"])
        self.assertAlmostEqual(verified.baseline_comparison["sharpe_delta"], 0.1, places=6)
        self.assertAlmostEqual(verified.baseline_comparison["delta_sharpe"], 0.1, places=6)
        self.assertAlmostEqual(verified.baseline_comparison["delta_calmar"], 0.2, places=6)
        self.assertAlmostEqual(verified.baseline_comparison["delta_return"], 1.0, places=6)
        self.assertFalse(verified.baseline_comparison["beats_baseline_objective"])
        self.assertTrue(verified.baseline_comparison["beats_baseline_guardrails"])


if __name__ == "__main__":
    unittest.main()
