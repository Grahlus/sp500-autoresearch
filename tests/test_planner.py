"""Tests for the Layer 2 planner package.

Tests cover:
  - plan_schema: validate_plan() for valid/invalid inputs
  - plan_store: save_plan(), load_current_plan(), archive_plan()
  - context_builder: build_planning_context() smoke test
  - llm_planner: call_minimax_planner() success/failure paths
"""
from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from planner.plan_schema import (
    ResearchPlan,
    is_plan_fresh,
    plan_to_dict,
    validate_plan,
)


def _valid_data(**overrides):
    now = datetime.now(UTC)
    data = {
        "plan_id": "test_plan_001",
        "created_at": now.isoformat(),
        "valid_until": (now + timedelta(hours=6)).isoformat(),
        "planner_name": "test_planner",
        "planner_model": "MiniMax-M2.7",
        "planner_provider": "minimax",
        "planning_window_hours": 6.0,
        "source_report_ids": [],
        "source_dashboard_timestamp": None,
        "family_budgets": {"momentum": 0.6, "superstock": 0.4},
        "template_budgets": {},
        "holdout_priorities": [],
        "new_idea_quota": None,
        "uncommon_idea_quota": None,
        "repeat_branch_cap": None,
        "structural_execution_lane_enabled": True,
        "structural_execution_lane_budget": None,
        "structural_execution_lane_min_candidates": None,
        "structural_execution_lane_repeat_cap": None,
        "stagnation_escape_active": False,
        "ml_probe_allowed": False,
        "rl_probe_allowed": False,
        "rationale": "Test rationale.",
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# plan_schema tests
# ---------------------------------------------------------------------------

class TestValidatePlan(unittest.TestCase):

    def test_valid_plan_round_trips(self):
        data = _valid_data()
        plan, reason = validate_plan(data)
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "ok")
        self.assertEqual(plan.plan_id, "test_plan_001")
        self.assertAlmostEqual(plan.planning_window_hours, 6.0)

    def test_missing_required_field_returns_error(self):
        data = _valid_data()
        del data["rationale"]
        plan, reason = validate_plan(data)
        self.assertIsNone(plan)
        self.assertIn("rationale", reason)

    def test_family_budgets_sum_exceeds_one(self):
        data = _valid_data(family_budgets={"momentum": 0.7, "superstock": 0.5})
        plan, reason = validate_plan(data)
        self.assertIsNone(plan)
        self.assertIn("family_budgets_sum", reason)

    def test_valid_until_before_created_at(self):
        now = datetime.now(UTC)
        data = _valid_data(
            created_at=(now + timedelta(hours=1)).isoformat(),
            valid_until=now.isoformat(),
        )
        plan, reason = validate_plan(data)
        self.assertIsNone(plan)
        self.assertIn("valid_until_not_after_created_at", reason)

    def test_invalid_cycle_mode_rejected(self):
        data = _valid_data(cycle_mode_override="super_mode")
        plan, reason = validate_plan(data)
        self.assertIsNone(plan)
        self.assertIn("cycle_mode_override_unknown", reason)

    def test_quota_out_of_range_rejected(self):
        data = _valid_data(new_idea_quota=1.5)
        plan, reason = validate_plan(data)
        self.assertIsNone(plan)
        self.assertIn("out_of_range", reason)

    def test_plan_to_dict_is_json_safe(self):
        data = _valid_data()
        plan, _ = validate_plan(data)
        serialized = plan_to_dict(plan)
        # Must be json-serialisable without error
        raw = json.dumps(serialized)
        roundtrip = json.loads(raw)
        self.assertEqual(roundtrip["plan_id"], "test_plan_001")

    def test_is_plan_fresh_true_for_future_valid_until(self):
        data = _valid_data()
        plan, _ = validate_plan(data)
        self.assertTrue(is_plan_fresh(plan))

    def test_is_plan_fresh_false_for_expired(self):
        now = datetime.now(UTC)
        data = _valid_data(
            created_at=(now - timedelta(hours=8)).isoformat(),
            valid_until=(now - timedelta(hours=1)).isoformat(),
        )
        plan, reason = validate_plan(data)
        self.assertIsNotNone(plan)  # valid schema
        self.assertFalse(is_plan_fresh(plan))


# ---------------------------------------------------------------------------
# plan_store tests
# ---------------------------------------------------------------------------

class TestPlanStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_cwd = pathlib.Path.cwd()
        import os
        os.chdir(self._tmp.name)

    def tearDown(self):
        import os
        os.chdir(str(self._orig_cwd))
        self._tmp.cleanup()

    def _make_plan(self, plan_id: str = "store_test_001") -> ResearchPlan:
        data = _valid_data(plan_id=plan_id)
        plan, _ = validate_plan(data)
        return plan

    def test_load_current_plan_no_file(self):
        from planner.plan_store import load_current_plan
        result = load_current_plan()
        self.assertFalse(result.loaded)
        self.assertEqual(result.reason, "no_file")

    def test_save_then_load_round_trips(self):
        from planner.plan_store import load_current_plan, save_plan
        plan = self._make_plan()
        save_plan(plan)
        result = load_current_plan()
        self.assertTrue(result.loaded)
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.plan.plan_id, "store_test_001")
        self.assertEqual(result.plan.planner_provider, "minimax")
        self.assertEqual(result.plan.planner_model, "MiniMax-M2.7")

    def test_archive_creates_file(self):
        from planner.plan_store import archive_plan
        plan = self._make_plan("archive_test_001")
        dest = archive_plan(plan)
        self.assertTrue(dest.exists())
        data = json.loads(dest.read_text())
        self.assertEqual(data["plan_id"], "archive_test_001")

    def test_load_expired_plan_returns_expired_reason(self):
        from planner.plan_store import load_current_plan, save_plan
        now = datetime.now(UTC)
        data = _valid_data(
            plan_id="expired_plan",
            created_at=(now - timedelta(hours=8)).isoformat(),
            valid_until=(now - timedelta(hours=1)).isoformat(),
        )
        plan, _ = validate_plan(data)
        save_plan(plan)
        result = load_current_plan()
        self.assertFalse(result.loaded)
        self.assertEqual(result.reason, "expired")


# ---------------------------------------------------------------------------
# context_builder smoke test
# ---------------------------------------------------------------------------

class TestContextBuilder(unittest.TestCase):

    def test_build_planning_context_returns_required_keys(self):
        """build_planning_context should always return the 6 required keys even with no data."""
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            context = build_planning_context(base_dir=tmpdir)

        required = {"timestamp_utc", "scorecards", "best_results",
                    "template_tracking", "recent_batches", "memory"}
        self.assertTrue(required.issubset(context.keys()))
        self.assertIsInstance(context["timestamp_utc"], str)
        self.assertIsInstance(context["best_results"], list)

    def test_build_planning_context_uses_scorecards_api_signature(self):
        """Regression test: context builder must not pass stale kwargs to scorecards."""
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("experiment_scorecards.build_family_scorecards", autospec=True, return_value={}):
                context = build_planning_context(base_dir=tmpdir)

        self.assertEqual(context["scorecards"], {})


# ---------------------------------------------------------------------------
# MiniMax planner tests
# ---------------------------------------------------------------------------

class TestMinimaxPlanner(unittest.TestCase):

    def test_returns_none_when_provider_raises(self):
        """call_minimax_planner must return None gracefully on provider failure."""
        from planner.llm_planner import call_minimax_planner
        with patch("planner.llm_planner.build_minimax_provider", side_effect=RuntimeError("no api key")):
            result = call_minimax_planner({})
        self.assertIsNone(result)

    def test_returns_none_on_invalid_json(self):
        from planner.llm_planner import call_minimax_planner

        mock_provider = MagicMock()
        mock_provider.create_message.return_value = MagicMock(text="not valid json at all")
        with patch("planner.llm_planner.build_minimax_provider", return_value=mock_provider):
            result = call_minimax_planner({})
        self.assertIsNone(result)

    def test_returns_plan_on_valid_response(self):
        from planner.llm_planner import call_minimax_planner

        plan_data = _valid_data(plan_id="cli_plan_001")
        mock_provider = MagicMock()
        mock_provider.create_message.return_value = MagicMock(text=json.dumps(plan_data))
        with patch("planner.llm_planner.build_minimax_provider", return_value=mock_provider):
            result = call_minimax_planner({})
        self.assertIsNotNone(result)
        self.assertEqual(result.plan_id, "cli_plan_001")
        self.assertEqual(result.planner_provider, "minimax")


# ---------------------------------------------------------------------------
# source_idea_ids round-trip tests
# ---------------------------------------------------------------------------

class TestSourceIdeaIds(unittest.TestCase):

    def test_source_idea_ids_default_is_empty_list(self):
        data = _valid_data()
        plan, reason = validate_plan(data)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.source_idea_ids, [])

    def test_source_idea_ids_populated_round_trips(self):
        data = _valid_data(source_idea_ids=["idea_001", "idea_002"])
        plan, reason = validate_plan(data)
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "ok")
        self.assertEqual(plan.source_idea_ids, ["idea_001", "idea_002"])

    def test_source_idea_ids_survives_plan_to_dict(self):
        data = _valid_data(source_idea_ids=["idea_x", "idea_y"])
        plan, _ = validate_plan(data)
        serialized = plan_to_dict(plan)
        self.assertEqual(serialized["source_idea_ids"], ["idea_x", "idea_y"])
        roundtrip, reason = validate_plan(serialized)
        self.assertEqual(reason, "ok")
        self.assertEqual(roundtrip.source_idea_ids, ["idea_x", "idea_y"])

    def test_structural_lane_fields_round_trip(self):
        data = _valid_data(
            structural_execution_lane_enabled=False,
            structural_execution_lane_budget=2,
            structural_execution_lane_min_candidates=1,
            structural_execution_lane_repeat_cap=3,
        )
        plan, reason = validate_plan(data)
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "ok")
        self.assertFalse(plan.structural_execution_lane_enabled)
        self.assertEqual(plan.structural_execution_lane_budget, 2)
        self.assertEqual(plan.structural_execution_lane_min_candidates, 1)
        self.assertEqual(plan.structural_execution_lane_repeat_cap, 3)
        serialized = plan_to_dict(plan)
        self.assertEqual(serialized["structural_execution_lane_enabled"], False)
        self.assertEqual(serialized["structural_execution_lane_budget"], 2)
        self.assertEqual(serialized["structural_execution_lane_min_candidates"], 1)
        self.assertEqual(serialized["structural_execution_lane_repeat_cap"], 3)


class TestPlannerCadence(unittest.TestCase):

    def test_should_replan_skips_when_plan_is_fresh(self):
        from planner.cadence import should_replan

        now = datetime.now(UTC)
        data = _valid_data(
            created_at=(now - timedelta(hours=1)).isoformat(),
            valid_until=(now + timedelta(hours=1)).isoformat(),
        )
        plan, _ = validate_plan(data)
        decision = should_replan(plan, min_replan_interval_hours=2.0, now=now)
        self.assertFalse(decision.should_replan)
        self.assertEqual(decision.reason, "plan_fresh_within_min_interval")

    def test_should_replan_triggers_for_old_plan(self):
        from planner.cadence import should_replan

        now = datetime.now(UTC)
        data = _valid_data(
            created_at=(now - timedelta(hours=7)).isoformat(),
            valid_until=(now + timedelta(hours=1)).isoformat(),
        )
        plan, _ = validate_plan(data)
        decision = should_replan(plan, min_replan_interval_hours=2.0, now=now)
        self.assertTrue(decision.should_replan)
        self.assertEqual(decision.reason, "plan_due")


# ---------------------------------------------------------------------------
# context_builder queued_ideas test
# ---------------------------------------------------------------------------

class TestContextBuilderQueuedIdeas(unittest.TestCase):

    def test_queued_ideas_key_present_in_context(self):
        """build_planning_context always returns a queued_ideas key."""
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            context = build_planning_context(base_dir=tmpdir, workspace_root=tmpdir)

        self.assertIn("queued_ideas", context)
        self.assertIsInstance(context["queued_ideas"], list)

    def test_queued_ideas_reads_json_files(self):
        """queued_ideas includes records from queues/ideas/."""
        import json
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_dir = pathlib.Path(tmpdir) / "queues" / "ideas"
            ideas_dir.mkdir(parents=True)
            idea = {
                "idea_id": "idea_test_001",
                "family": "momentum",
                "strategy_type": "classical",
                "hypothesis": "Test hypothesis",
                "source": "history_mining",
                "priority": 0.8,
                "status": "new",
                "timestamp_utc": "2026-04-19T00:00:00+00:00",
            }
            (ideas_dir / "idea_test_001.json").write_text(json.dumps(idea))

            context = build_planning_context(base_dir=tmpdir, workspace_root=tmpdir)

        self.assertEqual(len(context["queued_ideas"]), 1)
        self.assertEqual(context["queued_ideas"][0]["idea_id"], "idea_test_001")

    def test_queued_ideas_skips_consumed_records(self):
        """queued_ideas excludes ideas with status != new/pending."""
        import json
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_dir = pathlib.Path(tmpdir) / "queues" / "ideas"
            ideas_dir.mkdir(parents=True)
            consumed = {
                "idea_id": "idea_consumed",
                "family": "momentum",
                "strategy_type": "classical",
                "hypothesis": "Old idea",
                "source": "history_mining",
                "priority": 0.5,
                "status": "consumed",
                "timestamp_utc": "2026-04-01T00:00:00+00:00",
            }
            (ideas_dir / "idea_consumed.json").write_text(json.dumps(consumed))

            context = build_planning_context(base_dir=tmpdir, workspace_root=tmpdir)

        self.assertEqual(len(context["queued_ideas"]), 0)

    def test_web_research_status_and_provenance_are_visible(self):
        """Queued web-seeded ideas should keep provenance fields in planner context."""
        import json
        from planner.context_builder import build_planning_context

        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_dir = pathlib.Path(tmpdir) / "queues" / "ideas"
            status_dir = pathlib.Path(tmpdir) / "queues" / "web_research"
            ideas_dir.mkdir(parents=True)
            status_dir.mkdir(parents=True)
            idea = {
                "idea_id": "idea_web_001",
                "family": "momentum",
                "strategy_type": "classical",
                "hypothesis": "Web-seeded hypothesis",
                "source": "external_seed",
                "priority": 0.75,
                "status": "new",
                "timestamp_utc": "2026-04-19T00:00:00+00:00",
                "idea_source": "web_search",
                "source_idea_ids": ["paper_1"],
                "paper_title": "Recent momentum paper",
                "web_search_used": True,
            }
            (ideas_dir / "idea_web_001.json").write_text(json.dumps(idea))
            status = {
                "timestamp_utc": "2026-04-19T01:00:00+00:00",
                "backoff_state": "cooldown",
                "backoff_reason": "cooldown",
                "session_limit_hit": False,
                "last_attempt_at": "2026-04-19T01:00:00+00:00",
                "last_success_at": "2026-04-19T01:00:00+00:00",
                "last_failure_at": None,
                "next_retry_at": "2026-04-19T06:00:00+00:00",
                "queued_idea_count": 1,
                "queued_web_idea_count": 1,
                "web_search_available": True,
                "papers_found": 1,
            }
            (status_dir / "web_research_status.json").write_text(json.dumps(status))

            context = build_planning_context(base_dir=tmpdir, workspace_root=tmpdir)

        self.assertEqual(len(context["queued_ideas"]), 1)
        queued = context["queued_ideas"][0]
        self.assertEqual(queued["idea_source"], "web_search")
        self.assertEqual(queued["paper_title"], "Recent momentum paper")
        self.assertEqual(queued["source_idea_ids"], ["paper_1"])
        self.assertTrue(queued["web_search_used"])
        self.assertIn("web_research_status", context)
        self.assertEqual(context["web_research_status"]["backoff_state"], "cooldown")
        self.assertEqual(context["web_research_status"]["queued_web_idea_count"], 1)

    def test_backfill_source_idea_ids_uses_web_seeded_queue(self):
        """Planner backfill should consume web-seeded queued idea ids."""
        from planner.context_builder import build_planning_context
        from planner.run_planner import _backfill_source_idea_ids

        with tempfile.TemporaryDirectory() as tmpdir:
            ideas_dir = pathlib.Path(tmpdir) / "queues" / "ideas"
            ideas_dir.mkdir(parents=True)
            idea = {
                "idea_id": "idea_web_001",
                "family": "momentum",
                "strategy_type": "classical",
                "hypothesis": "Web-seeded hypothesis",
                "source": "external_seed",
                "priority": 0.75,
                "status": "new",
                "timestamp_utc": "2026-04-19T00:00:00+00:00",
                "idea_source": "web_search",
                "source_idea_ids": [],
                "paper_title": "Recent momentum paper",
                "web_search_used": True,
            }
            (ideas_dir / "idea_web_001.json").write_text(json.dumps(idea))

            context = build_planning_context(base_dir=tmpdir, workspace_root=tmpdir)
            data = _valid_data(source_idea_ids=[])
            plan, _ = validate_plan(data)
            filled = _backfill_source_idea_ids(plan, context["queued_ideas"])

        self.assertEqual(filled.source_idea_ids, ["idea_web_001"])


# ---------------------------------------------------------------------------
# run_planner source_idea_ids backfill test
# ---------------------------------------------------------------------------

class TestRunPlannerBackfill(unittest.TestCase):

    def test_backfill_source_idea_ids_fills_empty(self):
        """_backfill_source_idea_ids populates source_idea_ids when planner left it empty."""
        from planner.run_planner import _backfill_source_idea_ids

        data = _valid_data(source_idea_ids=[])
        plan, _ = validate_plan(data)
        queued = [{"idea_id": "idea_q1"}, {"idea_id": "idea_q2"}]
        filled = _backfill_source_idea_ids(plan, queued)
        self.assertEqual(filled.source_idea_ids, ["idea_q1", "idea_q2"])

    def test_backfill_does_not_overwrite_existing(self):
        """_backfill_source_idea_ids leaves source_idea_ids alone when already set."""
        from planner.run_planner import _backfill_source_idea_ids

        data = _valid_data(source_idea_ids=["idea_existing"])
        plan, _ = validate_plan(data)
        queued = [{"idea_id": "idea_other"}]
        filled = _backfill_source_idea_ids(plan, queued)
        self.assertEqual(filled.source_idea_ids, ["idea_existing"])

    def test_backfill_handles_empty_queue(self):
        """_backfill_source_idea_ids is a no-op when queue is empty."""
        from planner.run_planner import _backfill_source_idea_ids

        data = _valid_data(source_idea_ids=[])
        plan, _ = validate_plan(data)
        filled = _backfill_source_idea_ids(plan, [])
        self.assertEqual(filled.source_idea_ids, [])


class TestRunPlannerCli(unittest.TestCase):

    def test_run_planner_writes_valid_minimax_plan(self):
        from planner.run_planner import main

        with tempfile.TemporaryDirectory() as tmp:
            import os

            orig_cwd = pathlib.Path.cwd()
            os.chdir(tmp)
            try:
                experiments_dir = pathlib.Path(tmp) / "experiments"
                experiments_dir.mkdir(parents=True, exist_ok=True)
                plans_dir = pathlib.Path(tmp) / "plans"
                plans_dir.mkdir(parents=True, exist_ok=True)
                plan_data = _valid_data(
                    plan_id="minimax_plan_001",
                    created_at=datetime.now(UTC).isoformat(),
                    valid_until=(datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                )
                mock_provider = MagicMock()
                mock_provider.create_message.return_value = MagicMock(text=json.dumps(plan_data))
                with patch("planner.llm_planner.build_minimax_provider", return_value=mock_provider):
                    rc = main([
                        "--base-dir",
                        str(experiments_dir),
                        "--workspace-root",
                        tmp,
                        "--window-hours",
                        "2",
                        "--min-replan-interval-hours",
                        "2",
                    ])
                self.assertEqual(rc, 0)
                saved = json.loads((plans_dir / "current_plan.json").read_text())
                self.assertEqual(saved["planner_provider"], "minimax")
                self.assertEqual(saved["planner_model"], "MiniMax-M2.7")
                self.assertEqual(saved["plan_id"], "minimax_plan_001")
            finally:
                os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# executor integration tests (Layer 2 plan consumed by autonomous_runner)
# ---------------------------------------------------------------------------

class TestExecutorWithPlan(unittest.TestCase):
    """Tests that autonomous_runner loads and applies plans correctly."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_cwd = pathlib.Path.cwd()
        import os
        os.chdir(self._tmp.name)

    def tearDown(self):
        import os
        os.chdir(str(self._orig_cwd))
        self._tmp.cleanup()

    def _make_plan(self, **overrides):
        now = datetime.now(UTC)
        data = {
            "plan_id": "exec_test_plan",
            "created_at": now.isoformat(),
            "valid_until": (now + timedelta(hours=6)).isoformat(),
            "planner_name": "minimax_planner_v1",
            "planner_model": "MiniMax-M2.7",
            "planner_provider": "minimax",
            "planning_window_hours": 6.0,
            "source_report_ids": [],
            "source_idea_ids": ["idea_from_plan"],
            "source_dashboard_timestamp": None,
            "family_budgets": {"momentum": 0.7},
            "template_budgets": {},
            "holdout_priorities": [],
            "new_idea_quota": None,
            "uncommon_idea_quota": None,
            "repeat_branch_cap": None,
            "stagnation_escape_active": False,
            "ml_probe_allowed": False,
            "rl_probe_allowed": False,
            "rationale": "Test plan for executor integration.",
        }
        data.update(overrides)
        return data

    def test_executor_loads_valid_plan_and_logs_it(self, ):
        """Executor loads a valid plan and prints layer2_plan_loaded."""
        from planner.plan_store import save_plan
        plan, _ = validate_plan(self._make_plan())
        save_plan(plan)

        from autonomous_runner import _load_layer2_plan
        loaded_plan, reason = _load_layer2_plan()
        self.assertIsNotNone(loaded_plan)
        self.assertEqual(reason, "ok")
        self.assertEqual(loaded_plan.plan_id, "exec_test_plan")

    def test_executor_returns_none_for_missing_plan(self):
        """_load_layer2_plan returns (None, 'no_file') when plan file absent."""
        from autonomous_runner import _load_layer2_plan
        loaded_plan, reason = _load_layer2_plan()
        self.assertIsNone(loaded_plan)
        self.assertEqual(reason, "no_file")

    def test_executor_returns_none_for_expired_plan(self):
        """_load_layer2_plan returns (None, 'expired') for an expired plan."""
        from planner.plan_store import save_plan
        now = datetime.now(UTC)
        data = self._make_plan(
            created_at=(now - timedelta(hours=8)).isoformat(),
            valid_until=(now - timedelta(hours=1)).isoformat(),
        )
        plan, _ = validate_plan(data)
        save_plan(plan)

        from autonomous_runner import _load_layer2_plan
        loaded_plan, reason = _load_layer2_plan()
        self.assertIsNone(loaded_plan)
        self.assertEqual(reason, "expired")

    def test_apply_plan_hints_skipped_during_confirmation(self):
        """Structural lane hints survive confirmation; other advisory hints do not."""
        from planner.plan_schema import validate_plan as vp
        from autonomous_runner import _apply_plan_hints_to_kwargs
        from experiment_types import RuntimeDecision

        plan, _ = vp(
            self._make_plan(
                stagnation_escape_active=True,
                structural_execution_lane_enabled=True,
                structural_execution_lane_budget=2,
                structural_execution_lane_min_candidates=1,
                structural_execution_lane_repeat_cap=3,
            )
        )
        decision = RuntimeDecision(
            decision_id="d1",
            timestamp_utc="2026-04-19T00:00:00+00:00",
            status="active",
            selected_families=["momentum"],
            cycle_mode="confirmation",
            max_experiments=8,
            exploration_fraction=0.65,
            exploitation_fraction=0.35,
            family_budgets={"momentum": 8},
            large_search_mode=False,
            min_large_search_candidates=8,
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
            confirmation_required=True,
        )
        kwargs = {"stagnation_escape_batches": 3}
        _apply_plan_hints_to_kwargs(kwargs, plan, decision, 8)
        # structural lane hints survive confirmation, but non-lane hints do not
        self.assertTrue(kwargs["structural_execution_lane_enabled"])
        self.assertEqual(kwargs["structural_execution_lane_budget"], 2)
        self.assertEqual(kwargs["structural_execution_lane_min_candidates"], 1)
        self.assertEqual(kwargs["structural_execution_lane_repeat_cap"], 3)
        self.assertEqual(kwargs["stagnation_escape_batches"], 3)

    def test_apply_plan_hints_applies_stagnation_escape(self):
        """_apply_plan_hints_to_kwargs applies stagnation_escape_active when safe."""
        from planner.plan_schema import validate_plan as vp
        from autonomous_runner import _apply_plan_hints_to_kwargs
        from experiment_types import RuntimeDecision

        plan, _ = vp(self._make_plan(stagnation_escape_active=True))
        decision = RuntimeDecision(
            decision_id="d2",
            timestamp_utc="2026-04-19T00:00:00+00:00",
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
        kwargs = {"stagnation_escape_batches": 3}
        _apply_plan_hints_to_kwargs(kwargs, plan, decision, 24)
        self.assertEqual(kwargs["stagnation_escape_batches"], 1)

    def test_apply_plan_hints_merges_source_idea_ids(self):
        """_apply_plan_hints_to_kwargs merges plan source_idea_ids into kwargs."""
        from planner.plan_schema import validate_plan as vp
        from autonomous_runner import _apply_plan_hints_to_kwargs
        from experiment_types import RuntimeDecision

        plan, _ = vp(self._make_plan(source_idea_ids=["plan_idea_1", "plan_idea_2"]))
        decision = RuntimeDecision(
            decision_id="d3",
            timestamp_utc="2026-04-19T00:00:00+00:00",
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
        kwargs = {"source_idea_ids": ["existing_idea"]}
        _apply_plan_hints_to_kwargs(kwargs, plan, decision, 24)
        self.assertIn("plan_idea_1", kwargs["source_idea_ids"])
        self.assertIn("plan_idea_2", kwargs["source_idea_ids"])
        self.assertIn("existing_idea", kwargs["source_idea_ids"])

    def test_planner_failure_does_not_crash_load(self):
        """_load_layer2_plan returns (None, reason) even on import/parse errors."""
        import unittest.mock as mock
        with mock.patch("planner.plan_store.load_current_plan", side_effect=RuntimeError("disk error")):
            from autonomous_runner import _load_layer2_plan
            loaded_plan, reason = _load_layer2_plan()
        self.assertIsNone(loaded_plan)
        self.assertIn("disk error", reason)


# ---------------------------------------------------------------------------
# memory caps tests
# ---------------------------------------------------------------------------

class TestRuntimeMemoryCaps(unittest.TestCase):

    def test_no_cap_applied_when_rss_low(self):
        from autonomous_runner import _apply_runtime_memory_caps
        from experiment_types import RuntimeDecision
        from unittest.mock import patch

        decision = RuntimeDecision(
            decision_id="d_mem_low",
            timestamp_utc="2026-04-19T00:00:00+00:00",
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
        with patch("autonomous_runner.current_process_memory_kb",
                   return_value={"rss_kb": 4_000_000, "peak_rss_kb": 5_000_000}):
            batch_size, workers, caps = _apply_runtime_memory_caps(24, 6, decision)
        self.assertEqual(batch_size, 24)
        self.assertEqual(workers, 6)
        self.assertFalse(caps["applied"])


if __name__ == "__main__":
    unittest.main()
