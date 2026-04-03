import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autonomous_runner import main
from experiment_types import ExperimentResult, ExperimentSpec


class AutonomousRunnerTests(unittest.TestCase):
    def test_main_runs_single_round_and_writes_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ExperimentSpec(
                family="momentum",
                params={"LOOKBACK_WEEKS": 26},
                search_method="random",
                objective_name="wf_v1_score",
                batch_id="batch1",
                config_hash="abc123",
            )
            result = ExperimentResult(
                spec=spec,
                status="ok",
                score=1.0,
                metrics={"sharpe": 1.0, "sharpe_min": 0.5, "total_return_pct": 10.0, "trades_per_year": 20.0},
                robustness={"viable": True},
                artifacts={},
            )
            with patch("autonomous_runner.load_data", return_value={}), \
                patch("autonomous_runner.propose_initial_batch", return_value=[spec]), \
                patch("autonomous_runner.run_experiment_batch", return_value=[result]):
                rc = main(["--family", "momentum", "--n", "2", "--rounds", "1", "--seed", "7", "--base-dir", tmp])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(tmp) / "index.csv").exists())


if __name__ == "__main__":
    unittest.main()
