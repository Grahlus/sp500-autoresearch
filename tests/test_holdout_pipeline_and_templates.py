"""
Tests for:
  Part 1 — holdout pipeline bugs:
    - holdout_rationale fields present in compact_runtime_decision_payload
    - mixed_regime_clarification is in _TARGETED_HOLDOUT_TYPES
    - confirmation exhaustion guard spares named holdout types
    - _count_dedicated_holdout_runs filters correctly

  Part 2 — new structural momentum templates:
    - VOL_SCALE / DRIFT params exist in experiment_spaces
    - normalize_experiment_config pruning rules work
    - new IdeaTemplate entries exist in the library
    - agent.py globals exist

  Regression:
    - run.py / evaluate.py imports unchanged
    - existing momentum templates still validate
"""
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Part 1a: holdout_rationale in serialization
# ---------------------------------------------------------------------------

def _minimal_runtime_decision(**overrides):
    """Build a minimal valid RuntimeDecision for unit tests."""
    from experiment_types import RuntimeDecision
    base = dict(
        decision_id="test_decision",
        timestamp_utc="2026-04-15T00:00:00+00:00",
        status="ok",
        selected_families=["momentum"],
        cycle_mode="stagnation_escape",
        max_experiments=16,
        exploration_fraction=0.5,
        exploitation_fraction=0.5,
        family_budgets={"momentum": 16},
        large_search_mode=False,
        min_large_search_candidates=16,
        dashboard_report_id="test",
        latest_batch_overview={},
        latest_non_empty_batch={},
        best_overall=None,
        best_viable=None,
        best_baseline_beating=None,
        family_scorecards={},
        lineage_summary={},
        used_signals={},
        rationale={},
    )
    base.update(overrides)
    return RuntimeDecision(**base)


class TestHoldoutRationaleInPayload(unittest.TestCase):
    def test_compact_payload_includes_holdout_rationale(self):
        """compact_runtime_decision_payload must emit holdout_rationale and detail."""
        from experiment_runtime_decision import compact_runtime_decision_payload
        dec = _minimal_runtime_decision(
            holdout_rationale="holdout_pending_combined",
            holdout_rationale_detail="targeted_follow_up_type=mixed_regime_clarification",
        )
        payload = compact_runtime_decision_payload(dec, report_path=None)
        self.assertEqual(payload.get("holdout_rationale"), "holdout_pending_combined")
        self.assertEqual(
            payload.get("holdout_rationale_detail"),
            "targeted_follow_up_type=mixed_regime_clarification",
        )

    def test_compact_payload_includes_structural_novelty_budget(self):
        """compact_runtime_decision_payload must emit structural_novelty_budget."""
        from experiment_runtime_decision import compact_runtime_decision_payload
        dec = _minimal_runtime_decision(structural_novelty_budget=4)
        payload = compact_runtime_decision_payload(dec, report_path=None)
        self.assertEqual(payload.get("structural_novelty_budget"), 4)

    def test_holdout_rationale_default_is_no_winner(self):
        dec = _minimal_runtime_decision()
        self.assertEqual(dec.holdout_rationale, "no_winner")
        self.assertIsNone(dec.holdout_rationale_detail)

    def test_structural_novelty_budget_default_is_none(self):
        dec = _minimal_runtime_decision()
        self.assertIsNone(dec.structural_novelty_budget)


# ---------------------------------------------------------------------------
# Part 1b: mixed_regime_clarification in _TARGETED_HOLDOUT_TYPES
# ---------------------------------------------------------------------------

class TestTargetedHoldoutTypes(unittest.TestCase):
    def test_mixed_regime_clarification_in_set(self):
        from experiment_runtime_decision import _TARGETED_HOLDOUT_TYPES
        self.assertIn("mixed_regime_clarification", _TARGETED_HOLDOUT_TYPES)

    def test_mixed_regime_clarification_holdout_in_set(self):
        from experiment_runtime_decision import _TARGETED_HOLDOUT_TYPES
        self.assertIn("mixed_regime_clarification_holdout", _TARGETED_HOLDOUT_TYPES)

    def test_original_types_still_present(self):
        from experiment_runtime_decision import _TARGETED_HOLDOUT_TYPES
        for t in [
            "long_horizon_high_volatility_confirmation",
            "long_horizon_confirmation",
            "high_volatility_confirmation",
        ]:
            self.assertIn(t, _TARGETED_HOLDOUT_TYPES)

    def test_generic_type_not_in_set(self):
        from experiment_runtime_decision import _TARGETED_HOLDOUT_TYPES
        self.assertNotIn("generic_confirmation", _TARGETED_HOLDOUT_TYPES)
        self.assertNotIn("standard_confirmation", _TARGETED_HOLDOUT_TYPES)


# ---------------------------------------------------------------------------
# Part 1b: exhaustion guard spares named holdout types
# ---------------------------------------------------------------------------

class TestExhaustionGuardProtectsHoldouts(unittest.TestCase):
    """_targeted_is_holdout prevents confirmation_exhausted from firing."""

    def _run_exhaustion_logic(self, *, tfu_type, confirm_count=50, stagnation=60):
        """Replicate the guard logic extracted from _confirmation_batch_plan."""
        from experiment_runtime_decision import _TARGETED_HOLDOUT_TYPES
        targeted_follow_up = {"targeted_follow_up_type": tfu_type}
        _targeted_is_holdout = str(targeted_follow_up.get("targeted_follow_up_type") or "") in _TARGETED_HOLDOUT_TYPES
        escape_threshold = 3 * 5  # stagnation_escape_batches * 5
        confirmation_required = True
        confirmation_exhausted = (
            confirmation_required
            and confirm_count >= 45
            and stagnation >= escape_threshold
            and not _targeted_is_holdout
        )
        return confirmation_exhausted, _targeted_is_holdout

    def test_mixed_regime_not_exhausted(self):
        exhausted, is_holdout = self._run_exhaustion_logic(tfu_type="mixed_regime_clarification")
        self.assertTrue(is_holdout)
        self.assertFalse(exhausted, "mixed_regime_clarification should NOT be exhausted")

    def test_mixed_regime_holdout_not_exhausted(self):
        exhausted, is_holdout = self._run_exhaustion_logic(tfu_type="mixed_regime_clarification_holdout")
        self.assertTrue(is_holdout)
        self.assertFalse(exhausted)

    def test_long_horizon_not_exhausted(self):
        exhausted, is_holdout = self._run_exhaustion_logic(tfu_type="long_horizon_confirmation")
        self.assertFalse(exhausted)

    def test_generic_confirmation_can_be_exhausted(self):
        exhausted, is_holdout = self._run_exhaustion_logic(tfu_type="generic_confirmation")
        self.assertFalse(is_holdout)
        self.assertTrue(exhausted, "generic confirmation CAN be exhausted")

    def test_none_type_can_be_exhausted(self):
        exhausted, is_holdout = self._run_exhaustion_logic(tfu_type=None)
        self.assertFalse(is_holdout)
        self.assertTrue(exhausted)


# ---------------------------------------------------------------------------
# Part 1c: _count_dedicated_holdout_runs
# ---------------------------------------------------------------------------

class TestCountDedicatedHoldoutRuns(unittest.TestCase):
    def setUp(self):
        from experiment_scorecards import _count_dedicated_holdout_runs
        self.count = _count_dedicated_holdout_runs

    def test_empty_list_returns_zero(self):
        self.assertEqual(self.count([]), 0)

    def test_entry_with_batch_id_counts(self):
        entry = {"holdout_check_batch_id": "runtime_20260414_holdout_momentum"}
        self.assertEqual(self.count([entry]), 1)

    def test_entry_with_null_batch_id_does_not_count(self):
        entry = {"holdout_check_batch_id": None}
        self.assertEqual(self.count([entry]), 0)

    def test_entry_with_named_tfu_type_counts(self):
        entry = {"targeted_follow_up_type": "mixed_regime_clarification", "holdout_check_batch_id": None}
        self.assertEqual(self.count([entry]), 1)

    def test_entry_with_holdout_source_type_counts(self):
        entry = {"source_type": "targeted_holdout", "holdout_check_batch_id": None}
        self.assertEqual(self.count([entry]), 1)

    def test_generic_entry_without_batch_id_does_not_count(self):
        """A cycle where holdout was flagged but not dispatched should not count."""
        entry = {
            "holdout_check_batch_id": None,
            "holdout_check_required": True,
            "targeted_follow_up_type": "generic_confirmation",
            "source_type": "confirmation",
        }
        self.assertEqual(self.count([entry]), 0)

    def test_multiple_entries_mixed(self):
        entries = [
            {"holdout_check_batch_id": "batch_1"},          # counts
            {"holdout_check_batch_id": None},                # does not count
            {"targeted_follow_up_type": "long_horizon_confirmation"},  # counts
            {"holdout_check_batch_id": None, "source_type": "generic"},  # does not count
            {"source_type": "holdout"},                      # counts
        ]
        self.assertEqual(self.count(entries), 3)

    def test_non_dict_entries_are_skipped(self):
        self.assertEqual(self.count(["string_entry", 42, None]), 0)

    def test_50_generic_entries_count_zero(self):
        """Simulates the existing system state — 50 cycles flagged but never dispatched."""
        entries = [
            {"holdout_check_batch_id": None, "holdout_check_required": True}
            for _ in range(50)
        ]
        self.assertEqual(self.count(entries), 0)


# ---------------------------------------------------------------------------
# Part 2: new momentum parameters in search space
# ---------------------------------------------------------------------------

class TestNewMomentumParams(unittest.TestCase):
    def test_vol_scale_lookback_in_search_space(self):
        from experiment_spaces import SEARCH_SPACES
        self.assertIn("VOL_SCALE_LOOKBACK", SEARCH_SPACES["momentum"])

    def test_vol_scale_strength_in_search_space(self):
        from experiment_spaces import SEARCH_SPACES
        self.assertIn("VOL_SCALE_STRENGTH", SEARCH_SPACES["momentum"])

    def test_vol_scale_downside_in_search_space(self):
        from experiment_spaces import SEARCH_SPACES
        self.assertIn("VOL_SCALE_DOWNSIDE", SEARCH_SPACES["momentum"])

    def test_drift_lookback_weeks_in_search_space(self):
        from experiment_spaces import SEARCH_SPACES
        self.assertIn("DRIFT_LOOKBACK_WEEKS", SEARCH_SPACES["momentum"])

    def test_drift_min_slope_in_search_space(self):
        from experiment_spaces import SEARCH_SPACES
        self.assertIn("DRIFT_MIN_SLOPE", SEARCH_SPACES["momentum"])

    def test_vol_scale_defaults_when_disabled(self):
        """VOL_SCALE_LOOKBACK=0 must collapse STRENGTH and DOWNSIDE to defaults."""
        from experiment_spaces import normalize_experiment_config
        from experiment_spaces import get_family_default_config
        cfg = get_family_default_config("momentum")
        cfg["VOL_SCALE_LOOKBACK"] = 0
        cfg["VOL_SCALE_STRENGTH"] = 2.0  # non-default
        cfg["VOL_SCALE_DOWNSIDE"] = 1    # non-default
        result = normalize_experiment_config("momentum", cfg)
        self.assertEqual(result["VOL_SCALE_STRENGTH"], 1.0)
        self.assertEqual(result["VOL_SCALE_DOWNSIDE"], 0)

    def test_vol_scale_active_preserves_params(self):
        """VOL_SCALE_LOOKBACK=60 must keep STRENGTH and DOWNSIDE as provided."""
        from experiment_spaces import normalize_experiment_config
        from experiment_spaces import get_family_default_config
        cfg = get_family_default_config("momentum")
        cfg["VOL_SCALE_LOOKBACK"] = 60
        cfg["VOL_SCALE_STRENGTH"] = 2.0
        cfg["VOL_SCALE_DOWNSIDE"] = 1
        result = normalize_experiment_config("momentum", cfg)
        self.assertEqual(result["VOL_SCALE_STRENGTH"], 2.0)
        self.assertEqual(result["VOL_SCALE_DOWNSIDE"], 1)

    def test_drift_slope_collapsed_when_disabled(self):
        from experiment_spaces import normalize_experiment_config, get_family_default_config
        cfg = get_family_default_config("momentum")
        cfg["DRIFT_LOOKBACK_WEEKS"] = 0
        cfg["DRIFT_MIN_SLOPE"] = 0.02  # non-default (valid choice)
        result = normalize_experiment_config("momentum", cfg)
        self.assertEqual(result["DRIFT_MIN_SLOPE"], 0.0)

    def test_drift_slope_preserved_when_active(self):
        from experiment_spaces import normalize_experiment_config, get_family_default_config
        cfg = get_family_default_config("momentum")
        cfg["DRIFT_LOOKBACK_WEEKS"] = 26
        cfg["DRIFT_MIN_SLOPE"] = 0.02
        result = normalize_experiment_config("momentum", cfg)
        self.assertEqual(result["DRIFT_MIN_SLOPE"], 0.02)

    def test_existing_configs_still_valid_after_new_params(self):
        """Existing momentum configs without new params should still normalize cleanly."""
        from experiment_spaces import normalize_experiment_config
        old_config = {
            "LOOKBACK_WEEKS": 26, "SKIP_WEEKS": 3, "REBAL_WEEKS": 4,
            "TOP_PCT": 0.025, "MA_WEEKS": 20, "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.20, "STOP_PARABOLIC": 0.30, "INV_VOL_DAYS": 15,
            "MIN_HOLD_DAYS": 5, "FG_MIN": 10.0, "EXIT_PCT_RANK": 0.97,
            "RANK_EXIT_CONFIRM": None, "SIGNAL_TYPE": "momentum",
            "REVERSAL_FILTER_DAYS": 0, "DVOL_QUANTILE": 0.70, "CRASH_PROTECTION": 0,
        }
        result = normalize_experiment_config("momentum", old_config)
        self.assertIn("VOL_SCALE_LOOKBACK", result)
        self.assertEqual(result["VOL_SCALE_LOOKBACK"], 0)
        self.assertEqual(result["DRIFT_LOOKBACK_WEEKS"], 0)


# ---------------------------------------------------------------------------
# Part 2: new IdeaTemplate entries
# ---------------------------------------------------------------------------

class TestNewIdeaTemplates(unittest.TestCase):
    def _get_template_ids(self):
        from experiment_idea_library import list_idea_templates
        templates = list_idea_templates("momentum")
        return {t.template_id for t in templates}

    def test_vol_scaling_v1_template_exists(self):
        self.assertIn("momentum_vol_scaling_v1", self._get_template_ids())

    def test_vol_scaling_downside_v1_template_exists(self):
        self.assertIn("momentum_vol_scaling_downside_v1", self._get_template_ids())

    def test_drift_regime_v1_template_exists(self):
        self.assertIn("momentum_drift_regime_v1", self._get_template_ids())

    def test_drift_regime_strict_v1_template_exists(self):
        self.assertIn("momentum_drift_regime_strict_v1", self._get_template_ids())

    def test_vol_scaling_template_has_structural_novelty_source_type(self):
        from experiment_idea_library import list_idea_templates
        templates = {t.template_id: t for t in list_idea_templates("momentum")}
        t = templates["momentum_vol_scaling_v1"]
        self.assertEqual(t.source_type, "structural_novelty")

    def test_drift_regime_template_has_structural_novelty_source_type(self):
        from experiment_idea_library import list_idea_templates
        templates = {t.template_id: t for t in list_idea_templates("momentum")}
        t = templates["momentum_drift_regime_v1"]
        self.assertEqual(t.source_type, "structural_novelty")

    def test_vol_scaling_config_has_nondefault_params(self):
        from experiment_idea_library import list_idea_templates
        templates = {t.template_id: t for t in list_idea_templates("momentum")}
        t = templates["momentum_vol_scaling_v1"]
        self.assertGreater(t.config.get("VOL_SCALE_LOOKBACK", 0), 0)

    def test_drift_regime_config_has_nondefault_params(self):
        from experiment_idea_library import list_idea_templates
        templates = {t.template_id: t for t in list_idea_templates("momentum")}
        t = templates["momentum_drift_regime_v1"]
        self.assertGreater(t.config.get("DRIFT_LOOKBACK_WEEKS", 0), 0)

    def test_all_new_templates_validate_against_search_space(self):
        """New templates must produce valid configs through normalize_experiment_config."""
        from experiment_idea_library import list_idea_templates, materialize_template
        from experiment_spaces import normalize_experiment_config
        novel_ids = {
            "momentum_vol_scaling_v1", "momentum_vol_scaling_downside_v1",
            "momentum_drift_regime_v1", "momentum_drift_regime_strict_v1",
        }
        templates = [t for t in list_idea_templates("momentum") if t.template_id in novel_ids]
        self.assertEqual(len(templates), 4, "Expected 4 novel templates")
        for t in templates:
            with self.subTest(template=t.template_id):
                cfg = materialize_template("momentum", t)
                # Should not raise
                result = normalize_experiment_config("momentum", cfg)
                self.assertIsInstance(result, dict)
                self.assertIn("VOL_SCALE_LOOKBACK", result)
                self.assertIn("DRIFT_LOOKBACK_WEEKS", result)


# ---------------------------------------------------------------------------
# Part 2: agent.py globals exist
# ---------------------------------------------------------------------------

class TestAgentGlobals(unittest.TestCase):
    def test_vol_scale_lookback_exists(self):
        import agent
        self.assertTrue(hasattr(agent, "VOL_SCALE_LOOKBACK"))
        self.assertEqual(agent.VOL_SCALE_LOOKBACK, 0)

    def test_vol_scale_strength_exists(self):
        import agent
        self.assertTrue(hasattr(agent, "VOL_SCALE_STRENGTH"))
        self.assertEqual(agent.VOL_SCALE_STRENGTH, 1.0)

    def test_vol_scale_downside_exists(self):
        import agent
        self.assertTrue(hasattr(agent, "VOL_SCALE_DOWNSIDE"))
        self.assertEqual(agent.VOL_SCALE_DOWNSIDE, 0)

    def test_drift_lookback_weeks_exists(self):
        import agent
        self.assertTrue(hasattr(agent, "DRIFT_LOOKBACK_WEEKS"))
        self.assertEqual(agent.DRIFT_LOOKBACK_WEEKS, 0)

    def test_drift_min_slope_exists(self):
        import agent
        self.assertTrue(hasattr(agent, "DRIFT_MIN_SLOPE"))
        self.assertEqual(agent.DRIFT_MIN_SLOPE, 0.0)


# ---------------------------------------------------------------------------
# Part 2: new override keys in strategies/momentum.py
# ---------------------------------------------------------------------------

class TestMomentumStrategyOverrides(unittest.TestCase):
    def _get_overrides(self):
        import strategies.momentum as m
        import inspect, ast
        src = inspect.getsource(m._temporary_agent_config)
        # Parse out the overrides set from the source
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Set):
                return {elt.s for elt in node.elts if isinstance(elt, ast.Constant)}
        return set()

    def test_vol_scale_lookback_in_overrides(self):
        import strategies.momentum as m
        import inspect
        src = inspect.getsource(m._temporary_agent_config)
        self.assertIn("VOL_SCALE_LOOKBACK", src)

    def test_drift_lookback_weeks_in_overrides(self):
        import strategies.momentum as m
        import inspect
        src = inspect.getsource(m._temporary_agent_config)
        self.assertIn("DRIFT_LOOKBACK_WEEKS", src)


# ---------------------------------------------------------------------------
# Regression: imports and run.py/evaluate.py unaffected
# ---------------------------------------------------------------------------

class TestRegressionImports(unittest.TestCase):
    def test_experiment_runtime_decision_importable(self):
        import experiment_runtime_decision  # noqa

    def test_experiment_scorecards_importable(self):
        import experiment_scorecards  # noqa

    def test_experiment_spaces_importable(self):
        import experiment_spaces  # noqa

    def test_experiment_idea_library_importable(self):
        import experiment_idea_library  # noqa

    def test_experiment_types_importable(self):
        import experiment_types  # noqa

    def test_agent_importable(self):
        import agent  # noqa

    def test_strategies_momentum_importable(self):
        import strategies.momentum  # noqa


if __name__ == "__main__":
    unittest.main()
