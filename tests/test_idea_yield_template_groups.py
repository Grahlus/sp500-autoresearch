"""Tests for per-template-group idea-yield tracking.

Covers:
  - _STRUCTURAL_STRATEGY_TYPES constant
  - _template_group_key() classification
  - _aggregate_template_groups() aggregation
  - build_idea_yield_summary() "template_groups" key
  - FamilyScorecard.template_group_yield field
  - novelty_policy "template_group_yield_guidance" in runtime decision
"""
from __future__ import annotations

import types
import unittest
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs: Any):
    """Build a minimal pd.Series-like dict for testing _template_group_key."""
    import pandas as pd
    defaults = {
        "source_type": "",
        "strategy_type": "classical",
        "idea_kind": "new_idea_probe",
        "template_id": "unspecified",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def _make_frame(rows: list[dict[str, Any]]):
    """Build a minimal DataFrame for aggregation tests."""
    import pandas as pd
    defaults = {
        "strategy_family": "momentum",
        "source_type": "",
        "strategy_type": "classical",
        "idea_kind": "new_idea_probe",
        "template_id": "unspecified",
        "viable": False,
        "beats_baseline_objective": False,
        "beats_baseline_guardrails": False,
        "objective_score": 0.0,
        "timestamp_utc": "2026-01-01T00:00:00",
        "experiment_id": "exp001",
        "dead_zone_risk": 0.0,
        "status": "ok",
        "idea_source": "exploration",
    }
    filled = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r.update(row)
        r.setdefault("experiment_id", f"exp{i:03d}")
        filled.append(r)
    import pandas as pd
    return pd.DataFrame(filled)


# ---------------------------------------------------------------------------
# TestStructuralStrategyTypesConstant
# ---------------------------------------------------------------------------

class TestStructuralStrategyTypesConstant(unittest.TestCase):
    def test_constant_is_frozenset(self):
        from experiment_idea_yield import _STRUCTURAL_STRATEGY_TYPES
        self.assertIsInstance(_STRUCTURAL_STRATEGY_TYPES, frozenset)

    def test_contains_vol_regime_scaling(self):
        from experiment_idea_yield import _STRUCTURAL_STRATEGY_TYPES
        self.assertIn("vol_regime_scaling", _STRUCTURAL_STRATEGY_TYPES)

    def test_contains_drift_regime_conditioning(self):
        from experiment_idea_yield import _STRUCTURAL_STRATEGY_TYPES
        self.assertIn("drift_regime_conditioning", _STRUCTURAL_STRATEGY_TYPES)

    def test_does_not_contain_classical(self):
        from experiment_idea_yield import _STRUCTURAL_STRATEGY_TYPES
        self.assertNotIn("classical", _STRUCTURAL_STRATEGY_TYPES)

    def test_does_not_contain_ml(self):
        from experiment_idea_yield import _STRUCTURAL_STRATEGY_TYPES
        self.assertNotIn("ml", _STRUCTURAL_STRATEGY_TYPES)


# ---------------------------------------------------------------------------
# TestTemplateGroupKey
# ---------------------------------------------------------------------------

class TestTemplateGroupKey(unittest.TestCase):
    def _key(self, **kwargs):
        from experiment_idea_yield import _template_group_key
        return _template_group_key(_make_row(**kwargs))

    def test_classical_default(self):
        self.assertEqual(self._key(source_type="", strategy_type=""), "classical")

    def test_classical_explicit_strategy_type(self):
        self.assertEqual(self._key(source_type="", strategy_type="classical"), "classical")

    def test_classical_refinement_source(self):
        self.assertEqual(self._key(source_type="local_refinement", strategy_type=""), "classical")

    def test_vol_regime_via_strategy_type(self):
        self.assertEqual(
            self._key(source_type="", strategy_type="vol_regime_scaling"),
            "vol_regime_scaling",
        )

    def test_drift_regime_via_strategy_type(self):
        self.assertEqual(
            self._key(source_type="", strategy_type="drift_regime_conditioning"),
            "drift_regime_conditioning",
        )

    def test_structural_novelty_source_type_overrides(self):
        # source_type="structural_novelty" + strategy_type set → returns strategy_type
        self.assertEqual(
            self._key(source_type="structural_novelty", strategy_type="vol_regime_scaling"),
            "vol_regime_scaling",
        )

    def test_structural_novelty_source_no_strategy_type(self):
        # source_type="structural_novelty" but strategy_type empty → "structural_unknown"
        self.assertEqual(
            self._key(source_type="structural_novelty", strategy_type=""),
            "structural_unknown",
        )

    def test_exploration_source_is_classical(self):
        self.assertEqual(self._key(source_type="broad_exploration", strategy_type=""), "classical")


# ---------------------------------------------------------------------------
# TestAggregateTemplateGroups
# ---------------------------------------------------------------------------

class TestAggregateTemplateGroups(unittest.TestCase):
    def test_empty_frame_returns_empty_dict(self):
        import pandas as pd
        from experiment_idea_yield import _aggregate_template_groups
        result = _aggregate_template_groups(pd.DataFrame(), family="momentum")
        self.assertEqual(result, {})

    def test_single_classical_group(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"source_type": "broad_exploration", "strategy_type": "", "viable": True},
            {"source_type": "local_refinement", "strategy_type": "", "viable": False},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        self.assertIn("momentum::classical", result)
        self.assertNotIn("momentum::vol_regime_scaling", result)

    def test_structural_group_separated(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"source_type": "broad_exploration", "strategy_type": "classical"},
            {"source_type": "structural_novelty", "strategy_type": "vol_regime_scaling", "viable": True},
            {"source_type": "structural_novelty", "strategy_type": "vol_regime_scaling", "viable": False},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        self.assertIn("momentum::classical", result)
        self.assertIn("momentum::vol_regime_scaling", result)

    def test_composite_key_format(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"source_type": "structural_novelty", "strategy_type": "drift_regime_conditioning"},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        self.assertIn("momentum::drift_regime_conditioning", result)

    def test_group_metadata_fields(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"source_type": "broad_exploration", "strategy_type": "classical"},
            {"source_type": "broad_exploration", "strategy_type": "classical"},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        entry = result["momentum::classical"]
        self.assertEqual(entry["group_key"], "classical")
        self.assertEqual(entry["composite_key"], "momentum::classical")
        self.assertEqual(entry["family"], "momentum")
        self.assertEqual(entry["experiment_count"], 2)

    def test_experiment_count_per_group(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"strategy_type": "vol_regime_scaling"},
            {"strategy_type": "vol_regime_scaling"},
            {"strategy_type": "vol_regime_scaling"},
            {"strategy_type": "classical"},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        self.assertEqual(result.get("momentum::vol_regime_scaling", {}).get("experiment_count"), 3)
        self.assertEqual(result.get("momentum::classical", {}).get("experiment_count"), 1)

    def test_idea_quality_score_present(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([
            {"strategy_type": "vol_regime_scaling", "viable": True},
            {"strategy_type": "vol_regime_scaling", "viable": True},
        ])
        result = _aggregate_template_groups(frame, family="momentum")
        entry = result.get("momentum::vol_regime_scaling", {})
        self.assertIn("idea_quality_score", entry)
        self.assertIsInstance(entry["idea_quality_score"], float)

    def test_different_family_prefix(self):
        from experiment_idea_yield import _aggregate_template_groups
        frame = _make_frame([{"strategy_type": "vol_regime_scaling"}])
        result = _aggregate_template_groups(frame, family="superstock")
        self.assertIn("superstock::vol_regime_scaling", result)


# ---------------------------------------------------------------------------
# TestBuildIdeaYieldSummaryTemplateGroups
# ---------------------------------------------------------------------------

class TestBuildIdeaYieldSummaryTemplateGroups(unittest.TestCase):
    """Test that build_idea_yield_summary() emits a 'template_groups' key."""

    def _build(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import pandas as pd
        from experiment_idea_yield import build_idea_yield_summary

        frame = _make_frame(rows)
        # Override strategy_family to 'momentum' for all rows
        frame["strategy_family"] = "momentum"

        with patch("experiment_idea_yield.load_hot_experiment_index", return_value=frame), \
             patch("experiment_idea_yield.load_results_index", return_value=frame), \
             patch("experiment_idea_yield.load_research_memory", return_value={}), \
             patch("experiment_idea_yield.save_research_memory"):
            return build_idea_yield_summary(
                families=["momentum"],
                base_dir="experiments",
                persist_memory=False,
                use_hot_index=False,
            )

    def test_template_groups_key_present(self):
        result = self._build([{"strategy_type": "classical"}])
        self.assertIn("template_groups", result)

    def test_template_groups_is_dict(self):
        result = self._build([{"strategy_type": "classical"}])
        self.assertIsInstance(result["template_groups"], dict)

    def test_classical_group_in_template_groups(self):
        result = self._build([
            {"strategy_type": "classical"},
            {"strategy_type": "classical"},
        ])
        self.assertIn("momentum::classical", result["template_groups"])

    def test_structural_group_in_template_groups(self):
        result = self._build([
            {"strategy_type": "vol_regime_scaling", "source_type": "structural_novelty"},
            {"strategy_type": "classical"},
        ])
        self.assertIn("momentum::vol_regime_scaling", result["template_groups"])

    def test_family_dict_has_template_groups(self):
        result = self._build([{"strategy_type": "classical"}])
        family_data = result.get("families", {}).get("momentum", {})
        self.assertIn("template_groups", family_data)

    def test_empty_index_has_empty_template_groups(self):
        import pandas as pd
        from experiment_idea_yield import build_idea_yield_summary

        with patch("experiment_idea_yield.load_hot_experiment_index", return_value=pd.DataFrame()), \
             patch("experiment_idea_yield.load_results_index", return_value=pd.DataFrame()), \
             patch("experiment_idea_yield.load_research_memory", return_value={}), \
             patch("experiment_idea_yield.save_research_memory"):
            result = build_idea_yield_summary(
                families=["momentum"],
                base_dir="experiments",
                persist_memory=False,
                use_hot_index=False,
            )
        self.assertIn("template_groups", result)
        # May be empty dict or absent sub-keys — just must not error
        self.assertIsInstance(result["template_groups"], dict)


# ---------------------------------------------------------------------------
# TestFamilyScorecardTemplateGroupYieldField
# ---------------------------------------------------------------------------

class TestFamilyScorecardTemplateGroupYieldField(unittest.TestCase):
    def test_field_exists_on_dataclass(self):
        from experiment_scorecards import FamilyScorecard
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FamilyScorecard)}
        self.assertIn("template_group_yield", field_names)

    def test_field_default_is_none(self):
        from experiment_scorecards import FamilyScorecard
        import dataclasses
        for field in dataclasses.fields(FamilyScorecard):
            if field.name == "template_group_yield":
                self.assertIs(field.default, None)
                break

    def test_field_type_is_optional_dict(self):
        from experiment_scorecards import FamilyScorecard
        import dataclasses
        for field in dataclasses.fields(FamilyScorecard):
            if field.name == "template_group_yield":
                # The annotation should allow None
                ann = str(field.type)
                self.assertTrue("dict" in ann or "Dict" in ann, f"Unexpected type annotation: {ann}")
                break


# ---------------------------------------------------------------------------
# TestTemplateGroupYieldGuidanceInNoveltyPolicy
# ---------------------------------------------------------------------------

class TestTemplateGroupYieldGuidanceInNoveltyPolicy(unittest.TestCase):
    """Verify that novelty_policy gets 'template_group_yield_guidance'."""

    def _make_decision_payload(self, template_groups: dict[str, Any]) -> "RuntimeDecision":
        """Run build_runtime_decision with a mocked dashboard that includes template_groups."""
        from experiment_runtime_decision import build_runtime_decision, RuntimeDecisionInput

        # Minimal scorecard structure that passes all guards inside build_runtime_decision.
        minimal_scorecard = {
            "total_experiments": 10,
            "viable_rate": 0.2,
            "stagnation_experiments": 0,
            "dead_zone_density": 0.0,
            "duplicate_saturation": 0.0,
            "idea_yield_summary": {},
            "overfit_risk": 0.3,
            "best_objective_score": 0.4,
            "median_objective_score": 0.3,
            "exploration_budget_recommendation": 0.6,
            "exploitation_budget_recommendation": 0.4,
            "confirmation_history_count": 0,
            "holdout_history_count": 0,
            "winner_promotion_status": "not_promoted",
            "promotion_state": "unconfirmed",
            "promotion_blocked_pending_new_evidence": False,
            "holdout_check_required": False,
            "holdout_check_type": "generic_holdout",
            "holdout_check_status": "not_required",
            "holdout_check_outcome": "unproven",
            "holdout_check_scope": "generic",
            "validation_horizon_tags": [],
            "validation_regime_tags": [],
            "validation_scope": "unknown",
            "validation_confidence": 0.0,
            "validation_coverage": 0.0,
        }
        # counts.official_result_rows must be > 0 to pass the early fallback guard.
        mock_dashboard = {
            "generated_at_utc": "2026-01-01T00:00:00",
            "family_scorecards": {"momentum": minimal_scorecard},
            "counts": {"total": 10, "official_result_rows": 10},
            "top_overall": [],
            "top_viable": [],
            "top_baseline_beating": [],
            "latest_batch": {"batch_id": "b001", "timestamp_utc": "2026-01-01T00:00:00"},
            "latest_non_empty_batch": {"batch_id": "b001"},
            "lineage_summary": {},
            "idea_yield_summary": {"template_groups": template_groups},
        }
        mock_memory = {}
        with patch("experiment_runtime_decision.load_hot_summary_item", return_value=mock_dashboard), \
             patch("experiment_runtime_decision.load_research_memory", return_value=mock_memory), \
             patch("experiment_runtime_decision.save_research_memory"), \
             patch("experiment_runtime_decision._latest_batch_overview", return_value={}), \
             patch("experiment_runtime_decision.load_lineage_state_records", return_value={}):
            req = RuntimeDecisionInput(
                workspace_root=".",
                experiments_dir="experiments",
                strategy_families=["momentum"],
                max_experiments=24,
                seed=42,
            )
            decision = build_runtime_decision(req)
        return decision

    def test_template_group_yield_guidance_in_novelty_policy(self):
        decision = self._make_decision_payload({})
        rationale = decision.rationale or {}
        novelty_policy = rationale.get("novelty_policy") or {}
        self.assertIn("template_group_yield_guidance", novelty_policy)

    def test_template_group_budget_note_in_novelty_policy(self):
        decision = self._make_decision_payload({})
        rationale = decision.rationale or {}
        novelty_policy = rationale.get("novelty_policy") or {}
        self.assertIn("template_group_budget_note", novelty_policy)

    def test_empty_groups_gives_full_budget(self):
        decision = self._make_decision_payload({})
        rationale = decision.rationale or {}
        note = (rationale.get("novelty_policy") or {}).get("template_group_budget_note", "")
        # With no structural groups, budget note should mention no data
        self.assertIn("no template_group data yet", note)

    def test_low_quality_structural_group_reduces_budget(self):
        """Quality < 0.20 with ≥4 experiments should trigger 55% scale."""
        tg = {
            "momentum::vol_regime_scaling": {
                "group_key": "vol_regime_scaling",
                "family": "momentum",
                "experiment_count": 6,
                "idea_quality_score": 0.10,
                "idea_state": "stale",
                "idea_viable_rate": 0.0,
                "idea_retired_share": 0.5,
                "idea_promising_share": 0.0,
            }
        }
        decision = self._make_decision_payload(tg)
        rationale = decision.rationale or {}
        note = (rationale.get("novelty_policy") or {}).get("template_group_budget_note", "")
        self.assertIn("55%", note)

    def test_medium_quality_structural_group_reduces_budget_to_80(self):
        """Quality in [0.20, 0.45) with ≥4 experiments should trigger 80% scale."""
        tg = {
            "momentum::drift_regime_conditioning": {
                "group_key": "drift_regime_conditioning",
                "family": "momentum",
                "experiment_count": 5,
                "idea_quality_score": 0.30,
                "idea_state": "active",
                "idea_viable_rate": 0.1,
                "idea_retired_share": 0.2,
                "idea_promising_share": 0.1,
            }
        }
        decision = self._make_decision_payload(tg)
        rationale = decision.rationale or {}
        note = (rationale.get("novelty_policy") or {}).get("template_group_budget_note", "")
        self.assertIn("80%", note)

    def test_high_quality_structural_group_keeps_full_budget(self):
        """Quality ≥ 0.45 should not reduce budget."""
        tg = {
            "momentum::vol_regime_scaling": {
                "group_key": "vol_regime_scaling",
                "family": "momentum",
                "experiment_count": 8,
                "idea_quality_score": 0.55,
                "idea_state": "promising",
                "idea_viable_rate": 0.3,
                "idea_retired_share": 0.1,
                "idea_promising_share": 0.4,
            }
        }
        decision = self._make_decision_payload(tg)
        rationale = decision.rationale or {}
        note = (rationale.get("novelty_policy") or {}).get("template_group_budget_note", "")
        self.assertIn("full budget", note)

    def test_insufficient_experiments_skips_scaling(self):
        """Fewer than 4 experiments should not trigger budget scaling."""
        tg = {
            "momentum::vol_regime_scaling": {
                "group_key": "vol_regime_scaling",
                "family": "momentum",
                "experiment_count": 2,  # < 4 → no scaling
                "idea_quality_score": 0.05,
                "idea_state": "untested",
                "idea_viable_rate": 0.0,
                "idea_retired_share": 0.0,
                "idea_promising_share": 0.0,
            }
        }
        decision = self._make_decision_payload(tg)
        rationale = decision.rationale or {}
        note = (rationale.get("novelty_policy") or {}).get("template_group_budget_note", "")
        # No scaling should apply — note should still say "no template_group data yet"
        self.assertIn("no template_group data yet", note)

    def test_guidance_dict_has_expected_keys(self):
        tg = {
            "momentum::vol_regime_scaling": {
                "group_key": "vol_regime_scaling",
                "family": "momentum",
                "experiment_count": 4,
                "idea_quality_score": 0.2,
                "idea_state": "active",
                "idea_viable_rate": 0.1,
                "idea_retired_share": 0.1,
                "idea_promising_share": 0.1,
            }
        }
        decision = self._make_decision_payload(tg)
        rationale = decision.rationale or {}
        guidance = (rationale.get("novelty_policy") or {}).get("template_group_yield_guidance", {})
        self.assertIn("momentum::vol_regime_scaling", guidance)
        entry = guidance["momentum::vol_regime_scaling"]
        for key in ("group_key", "family", "experiment_count", "idea_quality_score", "idea_state"):
            self.assertIn(key, entry, f"missing key: {key}")


if __name__ == "__main__":
    unittest.main()
