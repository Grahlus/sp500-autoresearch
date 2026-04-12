import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.schemas import ProposalRecord, load_latest_pending_proposal_record, save_proposal_record
from autonomous_runner import main, parse_args
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

    def test_autonomous_defaults_are_widened_for_search(self):
        args = parse_args([])
        self.assertEqual(args.n, 24)
        self.assertEqual(args.max_workers, 6)
        self.assertAlmostEqual(args.exploration_fraction, 0.65)
        self.assertAlmostEqual(args.exploitation_fraction, 0.35)

    def test_main_passes_max_workers_to_batch_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch_request = BatchRequest(
                batch_id="batch1",
                timestamp_utc="2026-04-03T00:00:00+00:00",
                strategy_families=["momentum"],
                sampler_type="random",
                max_experiments=2,
                max_per_family=2,
                seed=7,
                persist=True,
                resume=True,
                max_workers=6,
            )
            batch_result = BatchResult(
                request=batch_request,
                status="completed",
                total_sampled=0,
                total_executed=0,
                total_skipped=0,
                total_failed=0,
                results=[],
            )
            with patch("autonomous_runner.run_batch_experiments", return_value=batch_result), patch(
                "autonomous_runner.build_batch_request", return_value=batch_request
            ) as mock_builder:
                rc = main(["--family", "momentum", "--n", "2", "--seed", "7", "--base-dir", tmp, "--max-workers", "6"])
            self.assertEqual(rc, 0)
            self.assertEqual(mock_builder.call_args.kwargs["max_workers"], 6)

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

    def test_main_can_consume_planning_agent_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_proposal_record(
                ProposalRecord(
                    proposal_id="queued1",
                    strategy_families=["momentum"],
                    source_idea_ids=["idea1"],
                    candidate_specs=[
                        {
                            "family": "momentum",
                            "params": {"LOOKBACK_WEEKS": 26},
                            "metadata": {"source_idea_ids": ["idea1"], "hypothesis": "queued proposal"},
                        }
                    ],
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    family_budget={"momentum": 1},
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                    planning_rationale={"planner": "test"},
                    analysis_provenance={"report_id": "analysis1", "batch_ids": ["batch0"]},
                ),
                workspace_root=tmp,
            )
            batch_request = BatchRequest(
                batch_id="queued_batch",
                timestamp_utc="2026-04-04T00:00:00+00:00",
                strategy_families=["momentum"],
                sampler_type="random",
                max_experiments=1,
                max_per_family=1,
                seed=7,
                persist=True,
                resume=True,
            )
            batch_result = BatchResult(
                request=batch_request,
                status="completed",
                total_sampled=1,
                total_executed=1,
                total_skipped=0,
                total_failed=0,
                results=[],
            )
            with patch("autonomous_runner.run_batch_experiments", return_value=batch_result), patch(
                "autonomous_runner.proposal_to_batch_request", return_value=batch_request
            ) as mock_converter, patch("autonomous_runner.build_batch_request") as mock_batch_builder, patch(
                "autonomous_runner.generate_next_round_proposal"
            ) as mock_generator:
                rc = main(["--family", "momentum", "--run-proposal", "--base-dir", str(Path(tmp) / "experiments")])
            self.assertEqual(rc, 0)
            self.assertTrue(mock_converter.called)
            queued_proposal = mock_converter.call_args.args[0]
            self.assertEqual(queued_proposal.request.source_idea_ids, ["idea1"])
            self.assertEqual(queued_proposal.request.source_batch_ids, ["batch0"])
            self.assertEqual(queued_proposal.candidate_metadata["momentum"][0]["source_idea_ids"], ["idea1"])
            self.assertEqual(queued_proposal.reasoning_summary["planning_rationale"], {"planner": "test"})
            self.assertEqual(queued_proposal.reasoning_summary["analysis_provenance"]["report_id"], "analysis1")
            self.assertFalse(mock_batch_builder.called)
            self.assertFalse(mock_generator.called)
            self.assertIsNone(load_latest_pending_proposal_record(tmp))

    def test_main_blocks_queued_proposal_when_quality_gate_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_proposal_record(
                ProposalRecord(
                    proposal_id="weak1",
                    strategy_families=["momentum"],
                    source_idea_ids=[],
                    candidate_specs=[],
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    family_budget={"momentum": 24},
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                    quality_report={
                        "status": "fail",
                        "execution_allowed": False,
                        "candidate_count": 0,
                        "requested": 24,
                        "min_viable_candidates": 12,
                        "shortfall_reasons": ["zero_selected_for_family"],
                    },
                ),
                workspace_root=tmp,
            )
            with patch("autonomous_runner.run_batch_experiments") as mock_runner, patch(
                "autonomous_runner.proposal_to_batch_request"
            ) as mock_converter:
                rc = main(["--family", "momentum", "--run-proposal", "--base-dir", str(Path(tmp) / "experiments")])
            self.assertEqual(rc, 0)
            self.assertFalse(mock_runner.called)
            self.assertFalse(mock_converter.called)
            self.assertEqual(load_latest_pending_proposal_record(tmp), None)

    def test_main_blocks_legacy_empty_queued_proposal_without_quality_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_proposal_record(
                ProposalRecord(
                    proposal_id="legacy_empty",
                    strategy_families=["momentum"],
                    source_idea_ids=[],
                    candidate_specs=[],
                    exploration_fraction=0.65,
                    exploitation_fraction=0.35,
                    family_budget={"momentum": 24},
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                ),
                workspace_root=tmp,
            )
            with patch("autonomous_runner.run_batch_experiments") as mock_runner, patch(
                "autonomous_runner.proposal_to_batch_request"
            ) as mock_converter:
                rc = main(["--family", "momentum", "--run-proposal", "--base-dir", str(Path(tmp) / "experiments")])
            self.assertEqual(rc, 0)
            self.assertFalse(mock_runner.called)
            self.assertFalse(mock_converter.called)
            self.assertEqual(load_latest_pending_proposal_record(tmp), None)

    def test_main_falls_back_to_internal_proposal_when_no_queue_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            proposal = ProposalResult(
                request=ProposalRequest(
                    proposal_id="proposal_fallback",
                    timestamp_utc="2026-04-04T00:00:00+00:00",
                    source_batch_ids=[],
                    strategy_families=["momentum"],
                    seed=7,
                    max_experiments=1,
                ),
                status="generated",
                candidate_configs={"momentum": [{"LOOKBACK_WEEKS": 26}]},
                reasoning_summary={"families": {"momentum": {"exploit_count": 1}}},
            )
            batch_request = BatchRequest(
                batch_id="fallback_batch",
                timestamp_utc="2026-04-04T00:00:00+00:00",
                strategy_families=["momentum"],
                sampler_type="random",
                max_experiments=1,
                max_per_family=1,
                seed=7,
                persist=True,
                resume=True,
            )
            batch_result = BatchResult(
                request=batch_request,
                status="completed",
                total_sampled=1,
                total_executed=1,
                total_skipped=0,
                total_failed=0,
                results=[],
            )
            with patch("autonomous_runner.generate_next_round_proposal", return_value=proposal) as mock_generator, patch(
                "autonomous_runner.proposal_to_batch_request", return_value=batch_request
            ), patch("autonomous_runner.run_batch_experiments", return_value=batch_result):
                rc = main(["--family", "momentum", "--run-proposal", "--base-dir", str(Path(tmp) / "experiments")])
            self.assertEqual(rc, 0)
            self.assertTrue(mock_generator.called)


if __name__ == "__main__":
    unittest.main()
