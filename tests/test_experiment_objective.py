import unittest

from experiment_objective import evaluate_objective, evaluate_robustness, is_experiment_viable


class ExperimentObjectiveTests(unittest.TestCase):
    def test_robustness_matches_run_constraints(self):
        metrics = {
            "sharpe": 0.7,
            "trades_per_year": 40,
            "sharpe_min": -0.8,
            "windows": [{"sharpe": 0.5}, {"sharpe": -0.1}, {"sharpe": 0.3}],
            "windows_beat_spy": "2/3",
        }
        robustness = evaluate_robustness(metrics)
        self.assertTrue(robustness["viable"])
        self.assertEqual(robustness["negative_windows"], 1)

    def test_non_viable_when_too_many_negative_windows(self):
        metrics = {
            "sharpe": 0.7,
            "trades_per_year": 40,
            "sharpe_min": -0.8,
            "windows": [{"sharpe": -0.1}] * 5,
            "windows_beat_spy": "0/5",
        }
        self.assertFalse(is_experiment_viable(metrics))

    def test_objective_prefers_higher_sharpe(self):
        a = evaluate_objective({"sharpe": 1.0, "calmar": 1.0, "total_return": 10, "trades_per_year": 20, "max_drawdown": -10}, "momentum")
        b = evaluate_objective({"sharpe": 0.5, "calmar": 1.0, "total_return": 10, "trades_per_year": 20, "max_drawdown": -10}, "momentum")
        self.assertGreater(a, b)


if __name__ == "__main__":
    unittest.main()
