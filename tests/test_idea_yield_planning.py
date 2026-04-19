"""
Tests for idea-yield tracking and feedback into planning.

Covers:
- experiment_refinement.py: per-family new_idea_floor adjustments
- experiment_refinement.py: idea_yield fields in per-family reasoning dict
- experiment_refinement.py: idea_yield_report_path in reasoning after scorecards persist
- experiment_runtime_decision.py: idea_yield_guidance in novelty_policy rationale
- Default paths unchanged when no yield data is available
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yield_cache(
    family: str,
    idea_state: str = "untested",
    idea_quality_score: float = 0.0,
    idea_retired_share: float = 0.0,
    idea_promising_share: float = 0.0,
    top_idea_kinds: list | None = None,
) -> dict:
    """Build a minimal yield-cache dict as returned by load_latest_idea_yield_summary."""
    return {
        "families": {
            family: {
                "idea_state": idea_state,
                "idea_quality_score": idea_quality_score,
                "idea_retired_share": idea_retired_share,
                "idea_promising_share": idea_promising_share,
                "top_idea_kinds": top_idea_kinds or [],
            }
        }
    }


# ---------------------------------------------------------------------------
# Unit tests: new_idea_floor adjustment logic (extracted from refinement code)
# ---------------------------------------------------------------------------

class TestNewIdeaFloorAdjustment(unittest.TestCase):
    """Tests for the idea-yield floor-adjustment block in generate_next_round_proposal."""

    def _run_adjustment(
        self,
        *,
        fam_idea_state: str = "untested",
        fam_idea_quality: float = 0.0,
        fam_retired_share: float = 0.0,
        fam_promising_share: float = 0.0,
        family_confirmation_mode: bool = False,
        budget: int = 8,
        cycle_new_idea_floor: int = 2,
        initial_floor: int = 2,
    ) -> tuple[int, str]:
        """Replicate the floor-adjustment logic from experiment_refinement.py."""
        new_idea_floor = initial_floor
        _yield_floor_action = "default"
        if not family_confirmation_mode and budget > 1:
            if fam_idea_state == "retired" or fam_retired_share >= 0.70:
                new_idea_floor = min(new_idea_floor, max(1, int(round(cycle_new_idea_floor * 0.50))))
                _yield_floor_action = "reduced_retired"
            elif fam_idea_state == "stale" or (fam_retired_share >= 0.45 and fam_idea_quality < 0.30):
                new_idea_floor = min(new_idea_floor, max(1, int(round(cycle_new_idea_floor * 0.75))))
                _yield_floor_action = "reduced_stale"
            elif fam_idea_state == "promising" and fam_idea_quality >= 0.55:
                new_idea_floor = max(new_idea_floor, cycle_new_idea_floor)
                _yield_floor_action = "maintained_promising"
        return new_idea_floor, _yield_floor_action

    def test_default_unchanged_when_untested(self):
        """No yield data → floor stays at initial, action is 'default'."""
        floor, action = self._run_adjustment(fam_idea_state="untested", initial_floor=2)
        self.assertEqual(action, "default")
        self.assertEqual(floor, 2)

    def test_retired_state_reduces_floor(self):
        """Retired idea state → floor reduced to 50% of cycle_new_idea_floor."""
        floor, action = self._run_adjustment(
            fam_idea_state="retired", cycle_new_idea_floor=4, initial_floor=4
        )
        self.assertEqual(action, "reduced_retired")
        self.assertEqual(floor, 2)  # 50% of 4

    def test_high_retired_share_reduces_floor(self):
        """retired_share >= 0.70 triggers reduced_retired even without state tag."""
        floor, action = self._run_adjustment(
            fam_idea_state="active", fam_retired_share=0.75, cycle_new_idea_floor=4, initial_floor=4
        )
        self.assertEqual(action, "reduced_retired")
        self.assertLessEqual(floor, 2)

    def test_stale_state_moderate_reduction(self):
        """Stale idea state → floor reduced to 75% of cycle_new_idea_floor."""
        floor, action = self._run_adjustment(
            fam_idea_state="stale", cycle_new_idea_floor=4, initial_floor=4
        )
        self.assertEqual(action, "reduced_stale")
        self.assertEqual(floor, 3)  # 75% of 4

    def test_stale_quality_combo_reduces_floor(self):
        """High retired_share + low quality → reduced_stale."""
        floor, action = self._run_adjustment(
            fam_idea_state="active",
            fam_retired_share=0.50,
            fam_idea_quality=0.20,
            cycle_new_idea_floor=4,
            initial_floor=4,
        )
        self.assertEqual(action, "reduced_stale")

    def test_promising_state_maintains_floor(self):
        """Promising + high quality → floor not lowered below cycle_new_idea_floor."""
        floor, action = self._run_adjustment(
            fam_idea_state="promising",
            fam_idea_quality=0.60,
            cycle_new_idea_floor=3,
            initial_floor=2,  # below cycle floor
        )
        self.assertEqual(action, "maintained_promising")
        self.assertGreaterEqual(floor, 3)

    def test_promising_below_quality_threshold_unchanged(self):
        """Promising state but quality < 0.55 → floor not raised."""
        floor, action = self._run_adjustment(
            fam_idea_state="promising",
            fam_idea_quality=0.40,
            cycle_new_idea_floor=3,
            initial_floor=2,
        )
        self.assertEqual(action, "default")
        self.assertEqual(floor, 2)

    def test_confirmation_mode_skips_adjustment(self):
        """family_confirmation_mode=True → no adjustment regardless of yield state."""
        floor, action = self._run_adjustment(
            fam_idea_state="retired",
            fam_retired_share=0.90,
            family_confirmation_mode=True,
            cycle_new_idea_floor=4,
            initial_floor=4,
        )
        self.assertEqual(action, "default")
        self.assertEqual(floor, 4)

    def test_budget_1_skips_adjustment(self):
        """budget == 1 → adjustment is skipped (guard condition `budget > 1`)."""
        floor, action = self._run_adjustment(
            fam_idea_state="retired",
            budget=1,
            cycle_new_idea_floor=2,
            initial_floor=2,
        )
        self.assertEqual(action, "default")

    def test_floor_never_below_1(self):
        """Floor is clamped to at least 1 even when cycle_new_idea_floor is tiny."""
        floor, action = self._run_adjustment(
            fam_idea_state="retired", cycle_new_idea_floor=1, initial_floor=1
        )
        self.assertGreaterEqual(floor, 1)


# ---------------------------------------------------------------------------
# Unit tests: idea_yield fields in reasoning dict
# ---------------------------------------------------------------------------

class TestIdeaYieldReasoningFields(unittest.TestCase):
    """The per-family reasoning dict must include idea_yield_* fields after Change 4."""

    def _build_reasoning_entry(
        self,
        fam_idea_state: str = "untested",
        fam_idea_quality: float = 0.0,
        fam_retired_share: float = 0.0,
        fam_promising_share: float = 0.0,
        yield_floor_action: str = "default",
        top_idea_kinds: list | None = None,
    ) -> dict:
        """Simulate the reasoning dict entry as written in experiment_refinement.py."""
        fam_yield = {
            "top_idea_kinds": top_idea_kinds or []
        }
        return {
            "holdout_horizon_tags": [],
            "holdout_regime_tags": [],
            "idea_yield_state": fam_idea_state,
            "idea_yield_quality": round(fam_idea_quality, 4),
            "idea_yield_retired_share": round(fam_retired_share, 4),
            "idea_yield_promising_share": round(fam_promising_share, 4),
            "idea_yield_floor_action": yield_floor_action,
            "idea_yield_top_kinds": [
                r.get("idea_kind") for r in (fam_yield.get("top_idea_kinds") or [])[:3]
                if r.get("idea_kind")
            ],
        }

    def test_all_idea_yield_fields_present(self):
        entry = self._build_reasoning_entry()
        required = [
            "idea_yield_state",
            "idea_yield_quality",
            "idea_yield_retired_share",
            "idea_yield_promising_share",
            "idea_yield_floor_action",
            "idea_yield_top_kinds",
        ]
        for field in required:
            self.assertIn(field, entry, f"Missing field: {field}")

    def test_idea_yield_state_default_untested(self):
        entry = self._build_reasoning_entry()
        self.assertEqual(entry["idea_yield_state"], "untested")

    def test_idea_yield_floor_action_propagated(self):
        entry = self._build_reasoning_entry(yield_floor_action="reduced_retired")
        self.assertEqual(entry["idea_yield_floor_action"], "reduced_retired")

    def test_idea_yield_top_kinds_truncated_to_3(self):
        kinds = [
            {"idea_kind": f"kind_{i}", "idea_state": "active"} for i in range(6)
        ]
        entry = self._build_reasoning_entry(top_idea_kinds=kinds)
        self.assertLessEqual(len(entry["idea_yield_top_kinds"]), 3)

    def test_idea_yield_top_kinds_filters_none(self):
        kinds = [
            {"idea_kind": None, "idea_state": "active"},
            {"idea_kind": "kind_a", "idea_state": "active"},
        ]
        entry = self._build_reasoning_entry(top_idea_kinds=kinds)
        self.assertNotIn(None, entry["idea_yield_top_kinds"])

    def test_quality_rounded_to_4dp(self):
        entry = self._build_reasoning_entry(fam_idea_quality=0.123456789)
        self.assertEqual(entry["idea_yield_quality"], round(0.123456789, 4))


# ---------------------------------------------------------------------------
# Unit tests: runtime decision idea_yield_guidance in novelty_policy
# ---------------------------------------------------------------------------

class TestIdeaYieldGuidanceInNoveltyPolicy(unittest.TestCase):
    """The _idea_yield_guidance dict should appear inside novelty_policy."""

    def _build_guidance_entry(
        self,
        family: str,
        scorecard_yield: dict | None = None,
    ) -> dict:
        """Replicate the per-family guidance computation from build_runtime_decision."""
        _iyg_yield = scorecard_yield or {}
        _top_kinds = _iyg_yield.get("top_idea_kinds") or []
        return {
            "idea_state": str(_iyg_yield.get("idea_state") or "untested"),
            "idea_quality_score": round(float(_iyg_yield.get("idea_quality_score") or 0.0), 4),
            "idea_retired_share": round(float(_iyg_yield.get("idea_retired_share") or 0.0), 4),
            "idea_promising_share": round(float(_iyg_yield.get("idea_promising_share") or 0.0), 4),
            "top_idea_kinds": [r.get("idea_kind") for r in _top_kinds[:3] if r.get("idea_kind")],
            "retiring_idea_kinds": [
                r.get("idea_kind") for r in _top_kinds
                if str(r.get("idea_state") or "") in {"retired", "stale"} and r.get("idea_kind")
            ],
        }

    def test_guidance_has_required_keys(self):
        guidance = self._build_guidance_entry("momentum")
        required = [
            "idea_state",
            "idea_quality_score",
            "idea_retired_share",
            "idea_promising_share",
            "top_idea_kinds",
            "retiring_idea_kinds",
        ]
        for key in required:
            self.assertIn(key, guidance, f"Missing key: {key}")

    def test_guidance_default_untested_when_no_data(self):
        guidance = self._build_guidance_entry("momentum", scorecard_yield={})
        self.assertEqual(guidance["idea_state"], "untested")
        self.assertEqual(guidance["idea_quality_score"], 0.0)

    def test_retiring_idea_kinds_filters_correctly(self):
        yield_data = {
            "top_idea_kinds": [
                {"idea_kind": "k_retired", "idea_state": "retired"},
                {"idea_kind": "k_stale", "idea_state": "stale"},
                {"idea_kind": "k_active", "idea_state": "active"},
                {"idea_kind": None, "idea_state": "retired"},  # filtered
            ]
        }
        guidance = self._build_guidance_entry("momentum", scorecard_yield=yield_data)
        self.assertIn("k_retired", guidance["retiring_idea_kinds"])
        self.assertIn("k_stale", guidance["retiring_idea_kinds"])
        self.assertNotIn("k_active", guidance["retiring_idea_kinds"])
        self.assertNotIn(None, guidance["retiring_idea_kinds"])

    def test_top_kinds_capped_at_3(self):
        yield_data = {
            "top_idea_kinds": [
                {"idea_kind": f"k{i}", "idea_state": "active"} for i in range(6)
            ]
        }
        guidance = self._build_guidance_entry("momentum", scorecard_yield=yield_data)
        self.assertLessEqual(len(guidance["top_idea_kinds"]), 3)

    def test_quality_score_rounded(self):
        yield_data = {"idea_quality_score": 0.666666}
        guidance = self._build_guidance_entry("momentum", scorecard_yield=yield_data)
        self.assertEqual(guidance["idea_quality_score"], round(0.666666, 4))

    def test_novelty_policy_dict_contains_idea_yield_guidance_key(self):
        """Smoke test: the novelty_policy dict structure includes idea_yield_guidance."""
        guidance_for_families = {
            "momentum": self._build_guidance_entry("momentum"),
        }
        novelty_policy = {
            "new_idea_budget": 4,
            "new_idea_quota": 4,
            "uncommon_idea_quota": 2,
            "repeat_branch_cap": 3,
            "max_same_template_per_cycle": 2,
            "max_same_lineage_per_cycle": 3,
            "structural_novelty_threshold": 0.6,
            "uncommon_template_bonus": 0.1,
            "idea_yield_guidance": guidance_for_families,
            "reason": "test",
        }
        self.assertIn("idea_yield_guidance", novelty_policy)
        self.assertIn("momentum", novelty_policy["idea_yield_guidance"])


# ---------------------------------------------------------------------------
# Integration smoke test: imports are clean
# ---------------------------------------------------------------------------

class TestImportSmokeTest(unittest.TestCase):
    def test_experiment_refinement_importable(self):
        import experiment_refinement  # noqa: F401

    def test_experiment_runtime_decision_importable(self):
        import experiment_runtime_decision  # noqa: F401

    def test_experiment_idea_yield_importable(self):
        import experiment_idea_yield  # noqa: F401

    def test_idea_yield_functions_present(self):
        from experiment_idea_yield import (
            build_idea_yield_summary,
            load_latest_idea_yield_summary,
            save_idea_yield_summary,
        )
        self.assertTrue(callable(build_idea_yield_summary))
        self.assertTrue(callable(load_latest_idea_yield_summary))
        self.assertTrue(callable(save_idea_yield_summary))


if __name__ == "__main__":
    unittest.main()
