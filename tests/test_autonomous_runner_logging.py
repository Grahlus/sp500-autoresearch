from __future__ import annotations

import unittest

import pandas as pd

from autonomous_runner import (
    _format_compact_dict,
    _format_compact_list,
    _format_leaderboard_preview,
    _format_reasoning_summary,
)


class AutonomousRunnerLoggingTests(unittest.TestCase):
    def test_format_compact_list_truncates_with_count(self) -> None:
        text = _format_compact_list([f"id-{index}" for index in range(6)], max_items=3)

        self.assertIn("id-0", text)
        self.assertIn("... +3 more", text)
        self.assertLess(len(text), 80)

    def test_format_compact_dict_truncates_nested_values(self) -> None:
        text = _format_compact_dict(
            {
                "a": list(range(10)),
                "b": {"inner": list(range(10))},
                "c": "x" * 200,
            },
            max_items=2,
        )

        self.assertIn("a=[", text)
        self.assertIn("b={", text)
        self.assertIn("... +1 more keys", text)
        self.assertLess(len(text), 220)

    def test_format_reasoning_summary_is_bounded(self) -> None:
        text = _format_reasoning_summary(
            {
                "proposal_quality": {
                    "status": "ok",
                    "quality_score": 0.87,
                    "candidate_count": 12,
                    "requested": 24,
                    "min_viable_candidates": 8,
                    "shortfall_reasons": ["one", "two", "three", "four", "five"],
                },
                "runtime_decision": {
                    "decision_id": "runtime_123",
                    "cycle_mode": "confirmation",
                    "fallback_used": False,
                    "selected_families": [f"family-{index}" for index in range(6)],
                    "confirmation_required": True,
                    "holdout_check_required": False,
                },
                "structural_execution_lane": {
                    "active": True,
                    "planner_requested": True,
                    "effective_budget": 2,
                    "candidate_count": 4,
                    "executed_count": 1,
                    "skipped": False,
                    "reason": "structural lane executed within its protected budget slice and remained under deterministic caps",
                },
                "analysis_provenance": {
                    "batch_ids": [f"batch-{index}" for index in range(6)],
                    "source_idea_ids": [f"idea-{index}" for index in range(7)],
                },
                "layer2_plan_id": "plan_123",
            },
        )

        self.assertIn("quality[status=ok", text)
        self.assertIn("runtime[decision_id=runtime_123", text)
        self.assertIn("structural_lane[active=True", text)
        self.assertIn("batch_ids=6", text)
        self.assertIn("ideas=7", text)
        self.assertIn("layer2_plan_id=plan_123", text)
        self.assertLess(len(text), 700)

    def test_format_leaderboard_preview_limits_rows_and_columns(self) -> None:
        leaderboard = pd.DataFrame(
            [
                {
                    "experiment_id": "exp-1",
                    "strategy_family": "momentum",
                    "objective_score": 1.2345,
                    "sharpe": 0.98,
                    "viable": True,
                    "beats_baseline_objective": True,
                    "comparison_status": "beat",
                },
                {
                    "experiment_id": "exp-2",
                    "strategy_family": "superstock",
                    "objective_score": 0.75,
                    "sharpe": 0.12,
                    "viable": False,
                    "beats_baseline_objective": False,
                    "comparison_status": "miss",
                },
                {
                    "experiment_id": "exp-3",
                    "strategy_family": "momentum",
                    "objective_score": 0.5,
                    "sharpe": -0.3,
                    "viable": False,
                    "beats_baseline_objective": False,
                    "comparison_status": "miss",
                },
                {
                    "experiment_id": "exp-4",
                    "strategy_family": "superstock",
                    "objective_score": 0.3,
                    "sharpe": -0.8,
                    "viable": False,
                    "beats_baseline_objective": False,
                    "comparison_status": "miss",
                },
                {
                    "experiment_id": "exp-5",
                    "strategy_family": "momentum",
                    "objective_score": 0.1,
                    "sharpe": -1.2,
                    "viable": False,
                    "beats_baseline_objective": False,
                    "comparison_status": "miss",
                },
                {
                    "experiment_id": "exp-6",
                    "strategy_family": "superstock",
                    "objective_score": 0.0,
                    "sharpe": -2.0,
                    "viable": False,
                    "beats_baseline_objective": False,
                    "comparison_status": "miss",
                },
            ]
        )

        text = _format_leaderboard_preview(leaderboard, max_rows=3)

        self.assertIn("leaderboard_preview top=3/6 cols=6", text)
        self.assertIn("experiment_id='exp-1'", text)
        self.assertIn("strategy_family='momentum'", text)
        self.assertIn("objective_score=1.234", text)
        self.assertNotIn("exp-6", text)
        self.assertIn("... +3 more rows", text)


if __name__ == "__main__":
    unittest.main()
