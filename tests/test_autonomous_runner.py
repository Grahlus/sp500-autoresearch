import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autonomous_runner import main
from experiment_types import BatchRequest, BatchResult, ExperimentResult, ExperimentSpec, ProposalRequest, ProposalResult


class AutonomousRunnerTests(unittest.TestCase):
    def test_main_runs_single_round_and_writes_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ExperimentResult(
                spec=ExperimentSpec(family="momentum", params={"LOOKBACK_WEEKS": 26}, config_hash="abc123", experiment_id="e1"),
                status="success",
                objective_score=1.0,
                metrics={"sharpe": 1.0, "calmar": 1.0, "total_return": 10.0, "trades_per_year": 20.0},
                robustness={"viable": True},
                artifacts={},
            )
            batch_result = BatchResult(
                request=BatchRequest(
                    batch_id="batch1",
                    timestamp_utc="2026-04-03T00:00:00+00:00",
                    strategy_families=["momentum"],
                    sampler_type="random",
                    max_experiments=2,
                    max_per_family=2,
                    seed=7,
                    persist=True,
                    resume=True,
                ),
                status="completed",
                total_sampled=1,
                total_executed=1,
                total_skipped=0,
                total_failed=0,
                results=[result],
                leaderboard_path=str(Path(tmp) / "batches" / "batch1" / "leaderboard.csv"),
                summary_path=str(Path(tmp) / "batches" / "batch1" / "summary.json"),
            )
            with patch("autonomous_runner.run_batch_experiments", return_value=batch_result), patch(
                "autonomous_runner.build_batch_request", return_value=batch_result.request
            ):
                rc = main(["--family", "momentum", "--n", "2", "--seed", "7", "--base-dir", tmp])
            self.assertEqual(rc, 0)
            self.assertTrue(batch_result.leaderboard_path.endswith("leaderboard.csv"))

    def test_main_can_generate_and_run_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            proposal = ProposalResult(
                request=ProposalRequest(
                    proposal_id="proposal1",
                    timestamp_utc="2026-04-03T00:00:00+00:00",
                    source_batch_ids=["b1"],
                    strategy_families=["momentum"],
                    seed=7,
                    max_experiments=2,
                ),
                status="generated",
                candidate_configs={"momentum": [{"LOOKBACK_WEEKS": 26}]},
                reasoning_summary={"families": {"momentum": {"exploit_count": 1}}},
                proposal_path=str(Path(tmp) / "proposal.json"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            batch_result = BatchResult(
                request=BatchRequest(
                    batch_id="batch1",
                    timestamp_utc="2026-04-03T00:00:00+00:00",
                    strategy_families=["momentum"],
                    sampler_type="random",
                    max_experiments=1,
                    max_per_family=1,
                    seed=7,
                    persist=True,
                    resume=True,
                ),
                status="completed",
                total_sampled=1,
                total_executed=1,
                total_skipped=0,
                total_failed=0,
                results=[],
                leaderboard_path=str(Path(tmp) / "leaderboard.csv"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            with patch("autonomous_runner.build_proposal_request", return_value=proposal.request), patch(
                "autonomous_runner.generate_next_round_proposal", return_value=proposal
            ), patch("autonomous_runner.proposal_to_batch_request", return_value=batch_result.request), patch(
                "autonomous_runner.run_batch_experiments", return_value=batch_result
            ):
                rc = main(["--family", "momentum", "--proposal-next", "--run-proposal", "--base-dir", tmp])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
