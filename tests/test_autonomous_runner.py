import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.schemas import ProposalRecord, load_latest_pending_proposal_record, save_proposal_record
from autonomous_runner import _apply_runtime_memory_caps, main, parse_args
from experiment_types import BatchRequest, BatchResult, ExperimentResult, ExperimentSpec, ProposalRequest, ProposalResult
from experiment_types import RuntimeDecision


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

    def test_main_uses_runtime_dashboard_decision_for_proposal_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = RuntimeDecision(
                decision_id="runtime_1",
                timestamp_utc="2026-04-12T00:00:00+00:00",
                status="active",
                selected_families=["momentum", "superstock"],
                cycle_mode="local_refinement",
                max_experiments=24,
                exploration_fraction=0.55,
                exploitation_fraction=0.45,
                family_budgets={"momentum": 18, "superstock": 6},
                large_search_mode=False,
                min_large_search_candidates=48,
                dashboard_report_id="dashboard_1",
                latest_batch_overview={"batch_id": "batch_1"},
                latest_non_empty_batch={"batch_id": "batch_1"},
                best_overall={"strategy_family": "momentum"},
                best_viable={"strategy_family": "momentum"},
                best_baseline_beating={"strategy_family": "momentum"},
                family_scorecards={"momentum": {"search_priority": 0.9}},
                lineage_summary={},
                used_signals={"best_viable": {"strategy_family": "momentum"}},
                rationale={"reason": "dashboard and scorecard evidence drove the runtime cycle mode and family budget split"},
                new_idea_quota=6,
                uncommon_idea_quota=2,
                repeat_branch_cap=4,
                max_same_template_per_cycle=3,
                max_same_lineage_per_cycle=3,
            )
            proposal = ProposalResult(
                request=ProposalRequest(
                    proposal_id="proposal_runtime",
                    timestamp_utc="2026-04-12T00:00:00+00:00",
                    source_batch_ids=["b1"],
                    strategy_families=["momentum", "superstock"],
                    seed=7,
                    max_experiments=24,
                ),
                status="generated",
                candidate_configs={
                    "momentum": [{"LOOKBACK_WEEKS": 26}],
                    "superstock": [{"N_TOP": 20}],
                },
                reasoning_summary={"families": {"momentum": {"exploit_count": 1}}},
                proposal_path=str(Path(tmp) / "proposal.json"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            batch_request = BatchRequest(
                batch_id="batch_runtime",
                timestamp_utc="2026-04-12T00:00:00+00:00",
                strategy_families=["momentum", "superstock"],
                sampler_type="random",
                max_experiments=2,
                max_per_family=1,
                seed=7,
                persist=True,
                resume=True,
            )
            batch_result = BatchResult(
                request=batch_request,
                status="completed",
                total_sampled=2,
                total_executed=2,
                total_skipped=0,
                total_failed=0,
                results=[],
                leaderboard_path=str(Path(tmp) / "leaderboard.csv"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            with patch("autonomous_runner.build_runtime_decision", return_value=decision), patch(
                "autonomous_runner.build_proposal_request", return_value=proposal.request
            ) as mock_builder, patch("autonomous_runner.generate_next_round_proposal", return_value=proposal), patch(
                "autonomous_runner.proposal_to_batch_request", return_value=batch_request
            ), patch("autonomous_runner.run_batch_experiments", return_value=batch_result) as mock_runner:
                rc = main(["--family", "all", "--proposal-next", "--run-proposal", "--base-dir", tmp])

            self.assertEqual(rc, 0)
            self.assertEqual(mock_builder.call_args.kwargs["exploration_fraction"], 0.55)
            self.assertEqual(mock_builder.call_args.kwargs["exploitation_fraction"], 0.45)
            self.assertEqual(mock_builder.call_args.kwargs["per_family_budgets"], {"momentum": 18, "superstock": 6})
            self.assertEqual(mock_builder.call_args.kwargs["new_idea_quota"], 6)
            self.assertEqual(mock_builder.call_args.kwargs["uncommon_idea_quota"], 2)
            self.assertEqual(mock_builder.call_args.kwargs["repeat_branch_cap"], 4)
            self.assertEqual(mock_builder.call_args.kwargs["max_same_template_per_cycle"], 3)
            self.assertEqual(mock_builder.call_args.kwargs["max_same_lineage_per_cycle"], 3)
            self.assertEqual(mock_builder.call_args.kwargs["structural_novelty_threshold"], 0.55)
            self.assertEqual(mock_builder.call_args.kwargs["min_large_search_candidates"], 48)
            self.assertEqual(mock_builder.call_args.kwargs["history_limit_per_family"], 12)
            request = mock_runner.call_args.args[0]
            self.assertIn("runtime_decision", request.proposal_metadata)
            self.assertEqual(request.proposal_metadata["runtime_decision"]["cycle_mode"], "local_refinement")
            self.assertEqual(request.proposal_metadata["runtime_decision"]["family_budgets"]["momentum"], 18)

    def test_main_uses_confirmation_runtime_decision_for_smaller_proposal_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = RuntimeDecision(
                decision_id="runtime_confirm",
                timestamp_utc="2026-04-12T00:00:00+00:00",
                status="active",
                selected_families=["momentum", "superstock"],
                cycle_mode="confirmation",
                max_experiments=24,
                exploration_fraction=0.75,
                exploitation_fraction=0.25,
                family_budgets={"momentum": 6, "superstock": 2},
                large_search_mode=False,
                min_large_search_candidates=8,
                dashboard_report_id="dashboard_2",
                latest_batch_overview={"batch_id": "batch_1"},
                latest_non_empty_batch={"batch_id": "batch_1"},
                best_overall={"strategy_family": "momentum"},
                best_viable={"strategy_family": "momentum"},
                best_baseline_beating={"strategy_family": "momentum"},
                family_scorecards={"momentum": {"search_priority": 0.9}},
                lineage_summary={},
                used_signals={"best_viable": {"strategy_family": "momentum"}},
                rationale={"reason": "dashboard and scorecard evidence drove the runtime cycle mode and family budget split"},
                promotion_state="unconfirmed",
                winner_family="momentum",
                winner_promotion_status="hold_for_confirmation",
                winner_exploitation_cap=0.25,
                confirmation_batch_requested=True,
                confirmation_required=True,
                confirmation_reason="winner needs a confirmation batch",
                confirmation_batch_id="runtime_confirm_momentum",
                planned_max_experiments=8,
                confirmation_family_budgets={"momentum": 6, "superstock": 2},
            )
            proposal = ProposalResult(
                request=ProposalRequest(
                    proposal_id="proposal_runtime_confirm",
                    timestamp_utc="2026-04-12T00:00:00+00:00",
                    source_batch_ids=["b1"],
                    strategy_families=["momentum", "superstock"],
                    seed=7,
                    max_experiments=8,
                ),
                status="generated",
                candidate_configs={
                    "momentum": [{"LOOKBACK_WEEKS": 26}],
                    "superstock": [{"N_TOP": 20}],
                },
                reasoning_summary={"families": {"momentum": {"exploit_count": 1}}},
                proposal_path=str(Path(tmp) / "proposal.json"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            batch_request = BatchRequest(
                batch_id="batch_runtime_confirm",
                timestamp_utc="2026-04-12T00:00:00+00:00",
                strategy_families=["momentum", "superstock"],
                sampler_type="random",
                max_experiments=2,
                max_per_family=1,
                seed=7,
                persist=True,
                resume=True,
            )
            batch_result = BatchResult(
                request=batch_request,
                status="completed",
                total_sampled=2,
                total_executed=2,
                total_skipped=0,
                total_failed=0,
                results=[],
                leaderboard_path=str(Path(tmp) / "leaderboard.csv"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            with patch("autonomous_runner.build_runtime_decision", return_value=decision), patch(
                "autonomous_runner.build_proposal_request", return_value=proposal.request
            ) as mock_builder, patch("autonomous_runner.generate_next_round_proposal", return_value=proposal), patch(
                "autonomous_runner.proposal_to_batch_request", return_value=batch_request
            ), patch("autonomous_runner.run_batch_experiments", return_value=batch_result) as mock_runner:
                rc = main(["--family", "all", "--proposal-next", "--run-proposal", "--base-dir", tmp])

            self.assertEqual(rc, 0)
            self.assertEqual(mock_builder.call_args.kwargs["max_experiments"], 8)
            self.assertEqual(mock_builder.call_args.kwargs["confirmation_state"], "unconfirmed")
            self.assertTrue(mock_builder.call_args.kwargs["confirmation_required"])
            self.assertEqual(mock_builder.call_args.kwargs["confirmation_reason"], "winner needs a confirmation batch")
            self.assertEqual(mock_builder.call_args.kwargs["confirmation_batch_id"], "runtime_confirm_momentum")
            self.assertEqual(mock_builder.call_args.kwargs["confirmation_focus_family"], "momentum")
            self.assertEqual(mock_builder.call_args.kwargs["confirmation_batch_experiments"], 8)
            self.assertEqual(mock_builder.call_args.kwargs["per_family_budgets"], {"momentum": 6, "superstock": 2})
            self.assertEqual(mock_builder.call_args.kwargs["history_limit_per_family"], 8)
            request = mock_runner.call_args.args[0]
            self.assertIn("runtime_decision", request.proposal_metadata)
            self.assertEqual(request.proposal_metadata["runtime_decision"]["promotion_state"], "unconfirmed")
            self.assertEqual(request.proposal_metadata["runtime_decision"]["planned_max_experiments"], 8)

    def test_main_caps_workers_for_confirmation_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            decision = RuntimeDecision(
                decision_id="runtime_confirm_cap",
                timestamp_utc="2026-04-12T00:00:00+00:00",
                status="active",
                selected_families=["momentum"],
                cycle_mode="confirmation",
                max_experiments=24,
                exploration_fraction=0.75,
                exploitation_fraction=0.25,
                family_budgets={"momentum": 8},
                large_search_mode=False,
                min_large_search_candidates=8,
                dashboard_report_id="dashboard_2",
                latest_batch_overview={"batch_id": "batch_1"},
                latest_non_empty_batch={"batch_id": "batch_1"},
                best_overall={"strategy_family": "momentum"},
                best_viable={"strategy_family": "momentum"},
                best_baseline_beating={"strategy_family": "momentum"},
                family_scorecards={"momentum": {"search_priority": 0.9}},
                lineage_summary={},
                used_signals={"best_viable": {"strategy_family": "momentum"}},
                rationale={"reason": "confirmation cycle"},
                promotion_state="unconfirmed",
                winner_family="momentum",
                winner_promotion_status="hold_for_confirmation",
                winner_exploitation_cap=0.25,
                confirmation_batch_requested=True,
                confirmation_required=True,
                confirmation_reason="winner needs a confirmation batch",
                confirmation_batch_id="runtime_confirm_momentum",
                planned_max_experiments=24,
                confirmation_family_budgets={"momentum": 8},
            )
            proposal = ProposalResult(
                request=ProposalRequest(
                    proposal_id="proposal_runtime_confirm_cap",
                    timestamp_utc="2026-04-12T00:00:00+00:00",
                    source_batch_ids=["b1"],
                    strategy_families=["momentum"],
                    seed=7,
                    max_experiments=8,
                ),
                status="generated",
                candidate_configs={"momentum": [{"LOOKBACK_WEEKS": 26}]},
                reasoning_summary={"families": {"momentum": {"exploit_count": 1}}},
                proposal_path=str(Path(tmp) / "proposal.json"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            batch_request = BatchRequest(
                batch_id="batch_confirm_cap",
                timestamp_utc="2026-04-12T00:00:00+00:00",
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
                leaderboard_path=str(Path(tmp) / "leaderboard.csv"),
                summary_path=str(Path(tmp) / "summary.json"),
            )
            with patch("autonomous_runner.build_runtime_decision", return_value=decision), patch(
                "autonomous_runner.build_proposal_request", return_value=proposal.request
            ), patch("autonomous_runner.generate_next_round_proposal", return_value=proposal), patch(
                "autonomous_runner.proposal_to_batch_request", return_value=batch_request
            ) as mock_converter, patch("autonomous_runner.run_batch_experiments", return_value=batch_result):
                rc = main(["--family", "all", "--proposal-next", "--run-proposal", "--base-dir", tmp, "--max-workers", "4", "--n", "24"])

            self.assertEqual(rc, 0)
            self.assertEqual(mock_converter.call_args.kwargs["max_workers"], 4)

    def test_runtime_memory_caps_downshift_on_high_rss(self):
        decision = RuntimeDecision(
            decision_id="runtime_1",
            timestamp_utc="2026-04-12T00:00:00+00:00",
            status="active",
            selected_families=["momentum"],
            cycle_mode="normal_exploration",
            max_experiments=24,
            exploration_fraction=0.65,
            exploitation_fraction=0.35,
            family_budgets={"momentum": 24},
            large_search_mode=False,
            min_large_search_candidates=48,
            dashboard_report_id=None,
            latest_batch_overview=None,
            latest_non_empty_batch=None,
            best_overall=None,
            best_viable=None,
            best_baseline_beating=None,
            family_scorecards={},
            lineage_summary={},
            used_signals={},
            rationale={},
        )
        with patch("autonomous_runner.current_process_memory_kb", return_value={"rss_kb": 15_500_000, "peak_rss_kb": 16_000_000}):
            batch_size, workers, memory_caps = _apply_runtime_memory_caps(24, 2, decision)
        self.assertEqual(batch_size, 16)
        self.assertEqual(workers, 1)
        self.assertTrue(memory_caps["applied"])
        self.assertEqual(memory_caps["reason"], "runtime_rss_pressure_elevated")

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
