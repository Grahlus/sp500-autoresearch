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

    def test_objective_penalizes_negative_sharpe_even_with_large_calmar_and_return(self):
        weak = evaluate_objective(
            {
                "sharpe": -0.156,
                "calmar": 11.989,
                "total_return": 36.7,
                "trades_per_year": 78.0,
                "max_drawdown": -31.62,
                "sharpe_min": -1.5,
                "windows": [{"sharpe": -0.2}] * 5,
            },
            "rl_bandit",
        )
        robust = evaluate_objective(
            {
                "sharpe": 0.706,
                "calmar": 2.141,
                "total_return": 12.74,
                "trades_per_year": 32.5,
                "max_drawdown": -21.67,
                "sharpe_min": -1.158,
                "windows": [{"sharpe": 0.2}] * 4,
            },
            "momentum",
        )
        self.assertLess(weak, 0.0)
        self.assertGreater(robust, weak)


if __name__ == "__main__":
    unittest.main()
