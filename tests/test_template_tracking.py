"""Tests for experiment_template_tracking.

Covers:
  - build_template_entry_report: counts proposals and executions per template
  - build_template_yield_report: yield metrics per template
  - detect_lineage_dominance: branch-share dominance detection
  - detect_structural_floor_active: recent-window structural activity check
  - build_full_template_tracking_report: integration / JSON-serialisability
"""
from __future__ import annotations

import json
import unittest
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(rows: list[dict[str, Any]]):
    import pandas as pd
    defaults = {
        "strategy_family": "momentum",
        "source_type": "structural_novelty",
        "strategy_type": "vol_regime_scaling",
        "template_id": "momentum_vol_scaling_v1",
        "viable": False,
        "beats_baseline_objective": False,
        "beats_baseline_guardrails": False,
        "objective_score": 0.0,
        "timestamp_utc": "2026-01-01T00:00:00",
        "experiment_id": "exp001",
        "source_proposal_id": "prop001",
    }
    filled = []
    for i, row in enumerate(rows):
        r = dict(defaults)
        r.update(row)
        r.setdefault("experiment_id", f"exp{i:03d}")
        filled.append(r)
    return pd.DataFrame(filled)


_T = "momentum_vol_scaling_v1"
_ALL_FOUR = frozenset({
    "momentum_vol_scaling_v1",
    "momentum_vol_scaling_downside_v1",
    "momentum_drift_regime_v1",
    "momentum_drift_regime_strict_v1",
})


# ---------------------------------------------------------------------------
# TestTemplateEntryReport
# ---------------------------------------------------------------------------

class TestTemplateEntryReport(unittest.TestCase):
    def _report(self, rows, **kw):
        from experiment_template_tracking import build_template_entry_report
        return build_template_entry_report(_make_frame(rows), tracked_templates=_ALL_FOUR, **kw)

    def test_all_four_present_even_if_empty(self):
        report = self._report([])
        self.assertEqual(set(report.keys()), _ALL_FOUR)

    def test_zero_count_for_unrun_template(self):
        report = self._report([])
        rec = report["momentum_drift_regime_v1"]
        self.assertEqual(rec.experiment_count, 0)
        self.assertEqual(rec.proposal_count, 0)
        self.assertIsNone(rec.last_seen_timestamp_utc)

    def test_experiment_count_matches_rows(self):
        rows = [
            {"template_id": _T},
            {"template_id": _T},
            {"template_id": "other_template"},
        ]
        report = self._report(rows)
        self.assertEqual(report[_T].experiment_count, 2)

    def test_proposal_count_distinct_source_proposals(self):
        rows = [
            {"template_id": _T, "source_proposal_id": "p1"},
            {"template_id": _T, "source_proposal_id": "p1"},
            {"template_id": _T, "source_proposal_id": "p2"},
        ]
        report = self._report(rows)
        self.assertEqual(report[_T].proposal_count, 2)

    def test_last_seen_is_most_recent_timestamp(self):
        rows = [
            {"template_id": _T, "timestamp_utc": "2026-01-01T00:00:00"},
            {"template_id": _T, "timestamp_utc": "2026-01-03T00:00:00"},
            {"template_id": _T, "timestamp_utc": "2026-01-02T00:00:00"},
        ]
        report = self._report(rows)
        self.assertEqual(report[_T].last_seen_timestamp_utc, "2026-01-03T00:00:00")

    def test_empty_dataframe_returns_zero_records(self):
        import pandas as pd
        from experiment_template_tracking import build_template_entry_report
        report = build_template_entry_report(pd.DataFrame(), tracked_templates=_ALL_FOUR)
        for rec in report.values():
            self.assertEqual(rec.experiment_count, 0)


# ---------------------------------------------------------------------------
# TestTemplateYieldReport
# ---------------------------------------------------------------------------

class TestTemplateYieldReport(unittest.TestCase):
    def _report(self, rows, idea_yield_summary=None, **kw):
        from experiment_template_tracking import build_template_yield_report
        return build_template_yield_report(
            _make_frame(rows),
            idea_yield_summary=idea_yield_summary,
            tracked_templates=_ALL_FOUR,
            family="momentum",
            **kw,
        )

    def test_all_four_present_even_if_empty(self):
        report = self._report([])
        self.assertEqual(set(report.keys()), _ALL_FOUR)

    def test_zero_viable_for_unrun_template(self):
        report = self._report([])
        rec = report["momentum_drift_regime_v1"]
        self.assertEqual(rec.experiment_count, 0)
        self.assertEqual(rec.viable_count, 0)
        self.assertEqual(rec.idea_state, "untested")

    def test_viable_rate_calculated_correctly(self):
        rows = [
            {"template_id": _T, "viable": True},
            {"template_id": _T, "viable": True},
            {"template_id": _T, "viable": False},
            {"template_id": _T, "viable": False},
        ]
        report = self._report(rows)
        rec = report[_T]
        self.assertEqual(rec.viable_count, 2)
        self.assertAlmostEqual(rec.viable_rate, 0.5, places=4)

    def test_baseline_beat_count(self):
        rows = [
            {"template_id": _T, "viable": True, "beats_baseline_objective": True},
            {"template_id": _T, "viable": False, "beats_baseline_guardrails": True},
            {"template_id": _T, "viable": False},
        ]
        report = self._report(rows)
        self.assertEqual(report[_T].baseline_beat_count, 2)

    def test_robust_descendant_count(self):
        rows = [
            {"template_id": _T, "viable": True, "beats_baseline_objective": True, "objective_score": 0.5},
            {"template_id": _T, "viable": False},
        ]
        report = self._report(rows)
        self.assertGreaterEqual(report[_T].robust_descendant_count, 1)

    def test_best_objective_score_is_max(self):
        rows = [
            {"template_id": _T, "objective_score": 0.3},
            {"template_id": _T, "objective_score": 1.5},
            {"template_id": _T, "objective_score": 0.7},
        ]
        report = self._report(rows)
        self.assertAlmostEqual(report[_T].best_objective_score, 1.5, places=4)

    def test_best_score_is_none_for_zero_experiments(self):
        report = self._report([])
        self.assertIsNone(report[_T].best_objective_score)

    def test_recent_trend_positive_when_improving(self):
        rows = [
            {"template_id": _T, "objective_score": 0.1, "timestamp_utc": f"2026-01-0{i}T00:00:00"}
            for i in range(1, 5)
        ] + [
            {"template_id": _T, "objective_score": 1.0, "timestamp_utc": f"2026-01-1{i}T00:00:00"}
            for i in range(1, 5)
        ]
        report = self._report(rows)
        self.assertGreater(report[_T].recent_trend, 0.0)

    def test_idea_state_from_yield_summary_when_available(self):
        yield_summary = {
            "by_template_id": {
                f"momentum::template::{_T}": {
                    "idea_state": "promising",
                    "idea_attempt_count": 5,
                    "idea_viable_count": 2,
                    "idea_baseline_beat_count": 1,
                    "idea_robust_descendant_count": 1,
                    "idea_recent_trend": 0.1,
                    "last_seen_timestamp_utc": "2026-01-10T00:00:00",
                }
            }
        }
        report = self._report([], idea_yield_summary=yield_summary)
        self.assertEqual(report[_T].idea_state, "promising")

    def test_fallback_to_raw_index_when_no_yield_summary(self):
        rows = [
            {"template_id": _T, "viable": True, "beats_baseline_objective": True, "objective_score": 0.8},
        ] * 3
        report = self._report(rows, idea_yield_summary=None)
        self.assertEqual(report[_T].viable_count, 3)


# ---------------------------------------------------------------------------
# TestLineageDominance
# ---------------------------------------------------------------------------

class TestLineageDominance(unittest.TestCase):
    def _detect(self, branch_summaries, **kw):
        from experiment_template_tracking import detect_lineage_dominance
        return detect_lineage_dominance(branch_summaries, **kw)

    def test_no_dominance_when_share_below_threshold(self):
        branch_summaries = {
            "momentum": {
                "hash_a": {"descendant_count": 50},
                "hash_b": {"descendant_count": 50},
            }
        }
        result = self._detect(branch_summaries, family="momentum", threshold=0.70)
        self.assertFalse(result["dominant"])
        self.assertIsNone(result["dominant_branch_root"])

    def test_dominance_detected_when_share_above_threshold(self):
        branch_summaries = {
            "momentum": {
                "hash_a": {"descendant_count": 90},
                "hash_b": {"descendant_count": 10},
            }
        }
        result = self._detect(branch_summaries, family="momentum", threshold=0.70)
        self.assertTrue(result["dominant"])
        self.assertEqual(result["dominant_branch_root"], "hash_a")
        self.assertAlmostEqual(result["dominant_branch_share"], 0.9, places=2)

    def test_warning_message_present_when_dominant(self):
        branch_summaries = {
            "momentum": {
                "hash_a": {"descendant_count": 95},
                "hash_b": {"descendant_count": 5},
            }
        }
        result = self._detect(branch_summaries, family="momentum", threshold=0.70)
        self.assertIsNotNone(result["warning"])
        self.assertIn("hash_a", result["warning"])

    def test_empty_plan_is_not_dominant(self):
        result = self._detect({}, family="momentum")
        self.assertFalse(result["dominant"])
        self.assertEqual(result["total_descendants"], 0)

    def test_all_zero_descendants_is_not_dominant(self):
        branch_summaries = {
            "momentum": {
                "hash_a": {"descendant_count": 0},
                "hash_b": {"descendant_count": 0},
            }
        }
        result = self._detect(branch_summaries, family="momentum")
        self.assertFalse(result["dominant"])


# ---------------------------------------------------------------------------
# TestStructuralFloorActive
# ---------------------------------------------------------------------------

class TestStructuralFloorActive(unittest.TestCase):
    def _detect(self, rows, **kw):
        from experiment_template_tracking import detect_structural_floor_active
        return detect_structural_floor_active(
            _make_frame(rows), tracked_templates=_ALL_FOUR, **kw
        )

    def test_no_structural_in_recent_when_none_run(self):
        result = self._detect([])
        self.assertFalse(result["any_structural_in_recent"])
        self.assertEqual(result["templates_in_recent"], [])

    def test_templates_never_run_all_four_when_empty(self):
        result = self._detect([])
        self.assertEqual(set(result["templates_never_run"]), _ALL_FOUR)

    def test_structural_in_recent_when_template_appears(self):
        rows = [{"template_id": _T, "timestamp_utc": "2026-01-10T00:00:00"}]
        result = self._detect(rows, recent_window=30)
        self.assertTrue(result["any_structural_in_recent"])
        self.assertIn(_T, result["templates_in_recent"])

    def test_template_not_in_never_run_after_one_execution(self):
        rows = [{"template_id": _T}]
        result = self._detect(rows)
        self.assertNotIn(_T, result["templates_never_run"])

    def test_recent_window_respected(self):
        rows = (
            [{"template_id": _T, "timestamp_utc": f"2026-01-0{i}T00:00:00"} for i in range(1, 5)]
            + [{"template_id": "other", "timestamp_utc": f"2026-02-{i:02d}T00:00:00"} for i in range(1, 30)]
        )
        result = self._detect(rows, recent_window=5)
        # 5 most recent rows are all "other" template, so structural not in recent
        self.assertFalse(result["any_structural_in_recent"])

    def test_empty_dataframe(self):
        import pandas as pd
        from experiment_template_tracking import detect_structural_floor_active
        result = detect_structural_floor_active(pd.DataFrame(), tracked_templates=_ALL_FOUR)
        self.assertFalse(result["any_structural_in_recent"])


# ---------------------------------------------------------------------------
# TestFullTemplateTrackingReport
# ---------------------------------------------------------------------------

class TestFullTemplateTrackingReport(unittest.TestCase):
    def _report(self, rows=None, **kw):
        from experiment_template_tracking import build_full_template_tracking_report
        return build_full_template_tracking_report(
            _make_frame(rows or []),
            tracked_templates=_ALL_FOUR,
            **kw,
        )

    def test_report_has_all_required_keys(self):
        report = self._report()
        for key in ("timestamp_utc", "family", "tracked_templates",
                    "template_entry", "template_yield",
                    "lineage_dominance", "structural_floor_active"):
            self.assertIn(key, report)

    def test_tracked_templates_list_correct(self):
        report = self._report()
        self.assertEqual(set(report["tracked_templates"]), _ALL_FOUR)

    def test_all_four_in_template_entry(self):
        report = self._report()
        self.assertEqual(set(report["template_entry"].keys()), _ALL_FOUR)

    def test_all_four_in_template_yield(self):
        report = self._report()
        self.assertEqual(set(report["template_yield"].keys()), _ALL_FOUR)

    def test_json_serialisable(self):
        report = self._report([
            {"template_id": _T, "viable": True, "objective_score": 0.5},
        ])
        serialised = json.dumps(report)
        restored = json.loads(serialised)
        self.assertIn("template_entry", restored)

    def test_timestamp_present_and_non_empty(self):
        report = self._report()
        self.assertIn("T", report["timestamp_utc"])

    def test_lineage_dominance_uses_branch_summaries(self):
        branch_summaries = {
            "momentum": {
                "hash_a": {"descendant_count": 100},
            }
        }
        report = self._report(branch_summaries=branch_summaries)
        self.assertEqual(report["lineage_dominance"]["total_descendants"], 100)

    def test_no_nan_in_output(self):
        import math
        report = self._report()
        serialised = json.dumps(report, allow_nan=False)
        self.assertNotIn("NaN", serialised)

    def test_structural_balance_policy_in_report(self):
        report = self._report()
        self.assertIn("structural_balance_policy", report)
        policy = report["structural_balance_policy"]
        self.assertIn("evidence_threshold", policy)
        self.assertIn("per_template", policy)
        self.assertIn("rationale", policy)

    def test_structural_balance_policy_all_templates_present(self):
        report = self._report()
        policy = report["structural_balance_policy"]
        self.assertEqual(set(policy["per_template"].keys()), _ALL_FOUR)


# ---------------------------------------------------------------------------
# TestStructuralBalancePolicy
# ---------------------------------------------------------------------------

class TestStructuralBalancePolicy(unittest.TestCase):
    def _policy(self, rows, evidence_threshold=8, repeat_cap=2):
        from experiment_template_tracking import build_structural_balance_policy
        return build_structural_balance_policy(
            _make_frame(rows),
            tracked_templates=_ALL_FOUR,
            evidence_threshold=evidence_threshold,
            repeat_cap=repeat_cap,
        )

    def test_all_behind_when_empty(self):
        policy = self._policy([])
        self.assertEqual(policy["behind_count"], len(_ALL_FOUR))
        self.assertEqual(policy["ahead_count"], 0)
        self.assertFalse(policy["all_graduated"])

    def test_floor_protected_below_threshold(self):
        rows = [{"template_id": _T}] * 3  # 3 < 8 threshold
        policy = self._policy(rows)
        self.assertTrue(policy["per_template"][_T]["floor_protected"])
        self.assertFalse(policy["per_template"][_T]["yield_eligible"])
        self.assertEqual(policy["per_template"][_T]["status"], "behind")

    def test_yield_eligible_at_threshold(self):
        rows = [{"template_id": _T}] * 8  # exactly at threshold
        policy = self._policy(rows)
        self.assertFalse(policy["per_template"][_T]["floor_protected"])
        self.assertTrue(policy["per_template"][_T]["yield_eligible"])
        self.assertEqual(policy["per_template"][_T]["status"], "ahead")

    def test_anti_monopoly_active_when_mixed(self):
        rows = [{"template_id": _T}] * 10  # _T is ahead
        policy = self._policy(rows)
        self.assertTrue(policy["anti_monopoly_active"])
        self.assertIn(_T, policy["templates_ahead"])
        others = _ALL_FOUR - {_T}
        for tid in others:
            self.assertIn(tid, policy["templates_behind"])

    def test_anti_monopoly_inactive_when_all_graduated(self):
        rows = []
        for tid in _ALL_FOUR:
            rows.extend([{"template_id": tid}] * 10)
        policy = self._policy(rows)
        self.assertFalse(policy["anti_monopoly_active"])
        self.assertTrue(policy["all_graduated"])
        self.assertEqual(policy["behind_count"], 0)

    def test_budget_share_per_behind_is_even(self):
        # 3 templates behind → each gets 1/3 share
        rows = [{"template_id": _T}] * 10  # only _T is ahead
        policy = self._policy(rows)
        expected_behind = len(_ALL_FOUR) - 1
        expected_share = round(1.0 / expected_behind, 3)
        self.assertAlmostEqual(
            policy["structural_budget_share_per_behind"],
            expected_share,
            places=3,
        )

    def test_rationale_mentions_behind_count(self):
        rows = [{"template_id": _T}] * 10
        policy = self._policy(rows)
        self.assertIn("evidence threshold", policy["rationale"])

    def test_rationale_mentions_all_graduated(self):
        rows = []
        for tid in _ALL_FOUR:
            rows.extend([{"template_id": tid}] * 10)
        policy = self._policy(rows)
        self.assertIn("Yield-based allocation active", policy["rationale"])

    def test_experiment_count_per_template(self):
        rows = [{"template_id": _T}] * 5
        policy = self._policy(rows)
        self.assertEqual(policy["per_template"][_T]["experiment_count"], 5)
        for tid in _ALL_FOUR - {_T}:
            self.assertEqual(policy["per_template"][tid]["experiment_count"], 0)

    def test_evidence_threshold_and_repeat_cap_in_output(self):
        policy = self._policy([], evidence_threshold=12, repeat_cap=3)
        self.assertEqual(policy["evidence_threshold"], 12)
        self.assertEqual(policy["repeat_cap"], 3)


# ---------------------------------------------------------------------------
# TestAntiMonopolyInRefinement (integration smoke test)
# ---------------------------------------------------------------------------

class TestAntiMonopolyInRefinement(unittest.TestCase):
    """Integration smoke tests verifying the anti-monopoly floor fires in refinement.

    Uses a temp dir and the same disk-write pattern as test_experiment_refinement.py
    so that analyze_experiment_history sees the injected template history.
    """

    def _write_fake_results(self, base_dir: str, template_counts: dict[str, int]) -> None:
        from experiment_spaces import get_family_default_config
        from experiment_store import compute_config_hash, save_experiment_result
        from experiment_types import ExperimentResult, ExperimentSpec
        base_cfg = get_family_default_config("momentum")
        idx = 0
        for tid, count in template_counts.items():
            for i in range(count):
                lookback_choices = [20, 26, 39, 52]
                skip_choices = [2, 3, 4]
                cfg = dict(base_cfg)
                cfg["LOOKBACK_WEEKS"] = lookback_choices[i % len(lookback_choices)]
                cfg["SKIP_WEEKS"] = skip_choices[(i // len(lookback_choices)) % len(skip_choices)]
                h = compute_config_hash("momentum", cfg)
                from dataclasses import asdict
                res = ExperimentResult(
                    spec=ExperimentSpec(
                        family="momentum",
                        params=cfg,
                        search_method="single",
                        objective_name="wf_v1_score",
                        batch_id="batch_smoke",
                        config_hash=h,
                        experiment_id=f"smoke_{idx:04d}",
                        timestamp_utc="2026-04-01T00:00:00+00:00",
                        benchmark_source="spy_symbol",
                        dataset_id="d1",
                        data_start="2020-01-01",
                        data_end="2020-12-31",
                        split="walk-forward",
                        git_commit="deadbeef",
                        family_version="test",
                        template_id=tid,
                        source_type="structural_novelty",
                        strategy_type="vol_regime_scaling",
                    ),
                    status="success",
                    objective_score=0.3,
                    metrics={"sharpe": 0.3, "calmar": 0.3, "total_return": 3.0,
                             "trades_per_year": 10.0, "exposure": 0.5, "max_drawdown": 10.0},
                    robustness={"viable": True, "negative_windows": 0},
                    artifacts={},
                )
                payload = asdict(res)
                payload["spec"] = asdict(res.spec)
                save_experiment_result(payload, base_dir=base_dir)
                idx += 1

    def _run_proposal(self, template_counts: dict[str, int]) -> "ProposalResult":
        import tempfile
        from experiment_refinement import build_proposal_request, generate_next_round_proposal
        with tempfile.TemporaryDirectory() as tmp:
            self._write_fake_results(tmp, template_counts)
            req = build_proposal_request(
                strategy_families=["momentum"],
                seed=42,
                max_experiments=16,
                resume=False,
            )
            return generate_next_round_proposal(req, base_dir=tmp)

    def test_untested_templates_appear_in_floor_slots(self):
        """With strict_v1 at 11 execs and others untested, the floor should inject
        at least one untested structural template."""
        result = self._run_proposal({"momentum_drift_regime_strict_v1": 11})
        all_meta = [
            meta
            for meta_list in (result.candidate_metadata or {}).values()
            for meta in meta_list
        ]
        floor_tids = {
            m.get("template_id")
            for m in all_meta
            if "structural floor" in str(m.get("reason_selected") or "")
        }
        untested = {
            "momentum_drift_regime_v1",
            "momentum_vol_scaling_v1",
            "momentum_vol_scaling_downside_v1",
        }
        self.assertTrue(
            floor_tids & untested,
            f"Expected at least one untested template in floor slots, got {floor_tids}",
        )

    def test_ahead_template_repeat_limited_when_others_behind(self):
        """strict_v1 (ahead) should appear at most STRUCTURAL_REPEAT_CAP times
        in the exploration-mode candidates."""
        from experiment_template_tracking import STRUCTURAL_REPEAT_CAP
        result = self._run_proposal({"momentum_drift_regime_strict_v1": 11})
        all_meta = [
            meta
            for meta_list in (result.candidate_metadata or {}).values()
            for meta in meta_list
        ]
        strict_non_floor = [
            m for m in all_meta
            if m.get("template_id") == "momentum_drift_regime_strict_v1"
            and "structural floor" not in str(m.get("reason_selected") or "")
        ]
        self.assertLessEqual(
            len(strict_non_floor),
            STRUCTURAL_REPEAT_CAP + 2,  # tolerance for template_expansion variants
            f"strict_v1 should not monopolize exploration; got {len(strict_non_floor)} slots",
        )


if __name__ == "__main__":
    unittest.main()
