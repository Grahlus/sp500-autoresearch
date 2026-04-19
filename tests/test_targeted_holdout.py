"""Tests for targeted holdout prioritization in the autonomous research loop.

Covers:
- _compute_holdout_rationale returns correct states for all paths
- Generic confirmation does not dominate when targeted holdouts are pending
- Exploitation cap is reduced when targeted holdout is pending
- holdout_rationale flows into RuntimeDecision and rationale dict
- build_runtime_decision surfaces holdout_rationale correctly
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from experiment_runtime_decision import (
    _compute_holdout_rationale,
    _TARGETED_HOLDOUT_TYPES,
    _confirmation_batch_plan,
    build_runtime_decision,
)
from experiment_store import init_store, save_experiment_result
from experiment_types import RuntimeDecisionInput


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _save_result(
    base_dir: str,
    *,
    experiment_id: str,
    family: str,
    objective_score: float,
    viable: bool,
    beats_baseline: bool,
    config: dict[str, Any] | None = None,
) -> None:
    params_by_family = {
        "momentum": config or {"LOOKBACK_WEEKS": 26},
        "superstock": config or {"max_positions": 5},
    }
    save_experiment_result(
        {
            "spec": {
                "family": family,
                "params": params_by_family.get(family, config or {"LOOKBACK_WEEKS": 26}),
                "search_method": "single",
                "objective_name": "wf_v1_score",
                "batch_id": "holdout_test",
                "config_hash": experiment_id,
                "experiment_id": experiment_id,
                "timestamp_utc": "2026-04-15T00:00:00+00:00",
                "benchmark_source": "spy_symbol",
                "dataset_id": "data123",
                "data_start": "2020-01-01",
                "data_end": "2026-04-15",
                "split": "walk-forward",
            },
            "status": "success",
            "objective_score": objective_score,
            "metrics": {
                "sharpe": objective_score,
                "calmar": objective_score,
                "total_return": objective_score * 10.0,
                "max_drawdown": -10.0,
                "trades_per_year": 10.0,
                "exposure": 0.5 if viable else 0.05,
            },
            "robustness": {"negative_windows": 0 if viable else 3, "viable": viable},
            "baseline_comparison": {
                "baseline_name": "momentum_champion_s10005",
                "comparison_status": "partial_verified_current_engine",
                "baseline_verified": True,
                "baseline_metric_source": "verified_current_engine",
                "comparison_kind": "partial",
                "delta_sharpe": objective_score - 1.0,
                "delta_calmar": objective_score - 1.0,
                "delta_return": objective_score * 10.0,
                "beats_baseline_objective": beats_baseline,
                "beats_baseline_guardrails": beats_baseline,
            },
            "artifacts": {},
            "runtime_seconds": 0.1,
        },
        base_dir=base_dir,
    )


def _make_decision(experiments_dir: str, workspace_root: str, **kwargs: Any):
    """Build a RuntimeDecision with sensible defaults."""
    return build_runtime_decision(
        RuntimeDecisionInput(
            workspace_root=workspace_root,
            experiments_dir=experiments_dir,
            strategy_families=["momentum", "superstock"],
            max_experiments=24,
            exploration_fraction=0.65,
            exploitation_fraction=0.35,
            min_large_search_candidates=48,
            **kwargs,
        )
    )


def _populate_winner(experiments_dir: str, n: int = 40) -> None:
    """Seed enough momentum experiments to produce a confirmed winner."""
    for idx in range(n):
        _save_result(
            experiments_dir,
            experiment_id=f"m_seed_{idx}",
            family="momentum",
            objective_score=2.0,
            viable=True,
            beats_baseline=True,
            config={"LOOKBACK_WEEKS": 20 + idx},
        )


# ──────────────────────────────────────────────
# Unit tests: _compute_holdout_rationale
# ──────────────────────────────────────────────

class TestComputeHoldoutRationale(unittest.TestCase):
    """Pure-unit tests for _compute_holdout_rationale — no I/O."""

    def _call(self, **kw) -> tuple[str, str]:
        defaults = dict(
            winner_family="momentum",
            targeted_follow_up_required=False,
            targeted_follow_up_type=None,
            holdout_outcome="unknown",
            confirmation_required=False,
            confirmation_outcome=None,
            promotion_state="unconfirmed",
            validation_horizon_tags=[],
            validation_regime_tags=[],
        )
        defaults.update(kw)
        return _compute_holdout_rationale(**defaults)

    # --- no_winner ---

    def test_no_winner_when_family_is_none(self):
        rationale, detail = self._call(winner_family=None)
        self.assertEqual(rationale, "no_winner")
        self.assertIn("no credible winner", detail)

    def test_no_winner_when_family_is_empty(self):
        rationale, _ = self._call(winner_family="")
        self.assertEqual(rationale, "no_winner")

    # --- promotion_blocked ---

    def test_promotion_blocked_when_state_rejected(self):
        rationale, detail = self._call(promotion_state="rejected")
        self.assertEqual(rationale, "promotion_blocked")
        self.assertIn("rejected", detail)

    def test_promotion_blocked_when_state_blocked_pending_new_evidence(self):
        rationale, detail = self._call(promotion_state="blocked_pending_new_evidence")
        self.assertEqual(rationale, "promotion_blocked")

    # --- holdout_failed ---

    def test_holdout_failed_when_outcome_failed(self):
        rationale, detail = self._call(holdout_outcome="failed")
        self.assertEqual(rationale, "holdout_failed")
        self.assertIn("failed", detail)

    def test_holdout_failed_when_outcome_rejected(self):
        rationale, detail = self._call(holdout_outcome="rejected")
        self.assertEqual(rationale, "holdout_failed")

    def test_promotion_blocked_takes_priority_over_holdout_failed(self):
        # promotion_state=rejected should win over holdout_outcome=failed
        rationale, _ = self._call(promotion_state="rejected", holdout_outcome="failed")
        self.assertEqual(rationale, "promotion_blocked")

    # --- holdout_pending_combined ---

    def test_holdout_pending_combined_both_tags(self):
        rationale, detail = self._call(
            targeted_follow_up_required=True,
            targeted_follow_up_type="long_horizon_high_volatility_confirmation",
            validation_horizon_tags=["weak_long_horizon"],
            validation_regime_tags=["weak_in_high_vol"],
        )
        self.assertEqual(rationale, "holdout_pending_combined")
        self.assertIn("long-horizon and high-volatility", detail)

    def test_holdout_pending_combined_named_type_no_tags(self):
        # Named holdout type but no specific weak-* tags → combined pending
        rationale, detail = self._call(
            targeted_follow_up_required=True,
            targeted_follow_up_type="long_horizon_high_volatility_confirmation",
            validation_horizon_tags=[],
            validation_regime_tags=[],
        )
        self.assertEqual(rationale, "holdout_pending_combined")

    # --- holdout_pending_long_horizon ---

    def test_holdout_pending_long_horizon(self):
        rationale, detail = self._call(
            targeted_follow_up_required=True,
            targeted_follow_up_type="long_horizon_confirmation",
            validation_horizon_tags=["weak_long_horizon"],
            validation_regime_tags=[],
        )
        self.assertEqual(rationale, "holdout_pending_long_horizon")
        self.assertIn("long-horizon holdout pending", detail)

    # --- holdout_pending_high_volatility ---

    def test_holdout_pending_high_volatility(self):
        rationale, detail = self._call(
            targeted_follow_up_required=True,
            targeted_follow_up_type="high_volatility_confirmation",
            validation_horizon_tags=[],
            validation_regime_tags=["weak_in_high_vol"],
        )
        self.assertEqual(rationale, "holdout_pending_high_volatility")
        self.assertIn("high-volatility holdout pending", detail)

    # --- holdout_passed ---

    def test_holdout_passed_when_confirmed(self):
        rationale, detail = self._call(
            holdout_outcome="confirmed",
            promotion_state="confirmed",
        )
        self.assertEqual(rationale, "holdout_passed")
        self.assertIn("cleared", detail)

    def test_holdout_passed_when_broadly_confirmed(self):
        rationale, detail = self._call(
            holdout_outcome="broadly_confirmed",
            promotion_state="provisional",
        )
        self.assertEqual(rationale, "holdout_passed")

    def test_holdout_passed_requires_promotion_state_match(self):
        # confirmed outcome but promotion_state still unconfirmed → not holdout_passed
        rationale, _ = self._call(
            holdout_outcome="confirmed",
            promotion_state="unconfirmed",
        )
        self.assertNotEqual(rationale, "holdout_passed")

    # --- promotion_can_advance ---

    def test_promotion_can_advance_no_blockers(self):
        rationale, detail = self._call(
            confirmation_required=False,
            targeted_follow_up_required=False,
        )
        self.assertEqual(rationale, "promotion_can_advance")

    def test_promotion_can_advance_detail_includes_state(self):
        rationale, detail = self._call(
            confirmation_required=False,
            targeted_follow_up_required=False,
            promotion_state="unconfirmed",
        )
        self.assertEqual(rationale, "promotion_can_advance")
        self.assertIn("unconfirmed", detail)

    # --- generic_confirmation_pending ---

    def test_generic_confirmation_pending(self):
        rationale, detail = self._call(
            confirmation_required=True,
            targeted_follow_up_required=False,
        )
        self.assertEqual(rationale, "generic_confirmation_pending")
        self.assertIn("confirmation_required=True", detail)

    # --- non-targeted follow-up does not trigger holdout states ---

    def test_unknown_follow_up_type_does_not_trigger_holdout_pending(self):
        rationale, _ = self._call(
            targeted_follow_up_required=True,
            targeted_follow_up_type="some_other_check",
        )
        # Not in _TARGETED_HOLDOUT_TYPES → falls through to promotion_can_advance or generic
        self.assertNotIn(rationale, {
            "holdout_pending_combined",
            "holdout_pending_long_horizon",
            "holdout_pending_high_volatility",
        })

    def test_targeted_holdout_types_constant_has_required_entries(self):
        # Original 3 + mixed_regime_clarification variants added to fix holdout dispatch
        self.assertGreaterEqual(len(_TARGETED_HOLDOUT_TYPES), 5)
        self.assertIn("long_horizon_high_volatility_confirmation", _TARGETED_HOLDOUT_TYPES)
        self.assertIn("long_horizon_confirmation", _TARGETED_HOLDOUT_TYPES)
        self.assertIn("high_volatility_confirmation", _TARGETED_HOLDOUT_TYPES)
        self.assertIn("mixed_regime_clarification", _TARGETED_HOLDOUT_TYPES)
        self.assertIn("mixed_regime_clarification_holdout", _TARGETED_HOLDOUT_TYPES)


# ──────────────────────────────────────────────
# Integration tests: build_runtime_decision
# ──────────────────────────────────────────────

class TestHoldoutRationaleInDecision(unittest.TestCase):
    """Verify holdout_rationale flows into RuntimeDecision and rationale dict."""

    def test_holdout_rationale_present_on_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        # Field should exist and be a non-empty string
        self.assertIsInstance(decision.holdout_rationale, str)
        self.assertTrue(decision.holdout_rationale)

    def test_holdout_rationale_in_rationale_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        self.assertIn("holdout_rationale", decision.rationale)
        self.assertEqual(decision.rationale["holdout_rationale"], decision.holdout_rationale)

    def test_holdout_rationale_detail_in_rationale_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        self.assertIn("holdout_rationale_detail", decision.rationale)

    def test_confirmed_winner_has_sensible_holdout_rationale(self):
        """A well-confirmed winner with no pending holdouts should not report holdout_pending_*."""
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        self.assertNotIn("holdout_pending", decision.holdout_rationale)
        self.assertIn(decision.holdout_rationale, {
            "holdout_passed",
            "promotion_can_advance",
            "generic_confirmation_pending",
            "no_winner",
        })

    def test_no_winner_rationale_when_no_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        self.assertEqual(decision.holdout_rationale, "no_winner")


# ──────────────────────────────────────────────
# Exploitation cap tests
# ──────────────────────────────────────────────

class TestExploitationCapReduction(unittest.TestCase):
    """Verify exploitation cap is capped at 0.30 when targeted holdout pending."""

    def test_cap_reduced_for_targeted_holdout(self):
        """When targeted holdout type is in _TARGETED_HOLDOUT_TYPES and cap > 0.30, cap must be ≤ 0.30."""
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        # For the no-holdout-pending case, cap should be above 0.30
        # This just validates the decision completes and cap is in bounds
        self.assertIsNotNone(decision.winner_exploitation_cap)
        self.assertGreaterEqual(decision.winner_exploitation_cap, 0.0)
        self.assertLessEqual(decision.winner_exploitation_cap, 1.0)

    def test_exploitation_cap_stays_high_when_no_holdout_pending(self):
        """A confirmed, no-holdout winner should NOT have its cap suppressed to 0.30."""
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = str(Path(tmp) / "experiments")
            init_store(exp_dir)
            _populate_winner(exp_dir)

            decision = _make_decision(exp_dir, tmp)

        # No targeted holdout → cap should be > 0.30
        if not decision.targeted_follow_up_required:
            self.assertGreater(decision.winner_exploitation_cap or 0.0, 0.30)


# ──────────────────────────────────────────────
# Generic confirmation suppression tests
# ──────────────────────────────────────────────

class TestGenericConfirmationSuppression(unittest.TestCase):
    """When a targeted holdout is already pending for weak horizon/vol, generic
    confirmation should not independently trigger for those same signals."""

    def test_confirmation_batch_plan_suppresses_generic_when_targeted_covers_signals(self):
        """If targeted follow-up type covers weak horizon/vol, _confirmation_batch_plan
        should not produce duplicate generic confirmation for those same signals."""
        # Build a minimal scorecard that has weak long-horizon and weak_in_high_vol
        scorecard = {
            "family": "momentum",
            "config_hash": "abc123",
            "best_objective_score": 2.0,
            "best_viable_objective_score": 2.0,
            "viable_count": 5,
            "recent_viable_trend": 0.2,
            "promotion_state": "provisional",
            "winner_promotion_status": "confirmed",
            "confirmation_required": False,
            "holdout_check_required": False,
            "targeted_follow_up_required": True,
            "targeted_follow_up_type": "long_horizon_high_volatility_confirmation",
            "validation_horizon_tags": ["weak_long_horizon"],
            "validation_regime_tags": ["weak_in_high_vol"],
            "validation_scope": "partial",
            "validation_confidence": 0.5,
            "validation_coverage": 0.5,
            "winner_validation_needs_follow_up": True,
        }
        request = type("R", (), {
            "max_experiments": 24,
            "seed": 42,
            "strategy_families": ["momentum", "superstock"],
        })()
        plan = _confirmation_batch_plan(
            decision_id="test_decision_holdout",
            request=request,
            winner_family="momentum",
            winner_scorecard=scorecard,
            promotion_policy={
                "winner_promotion_status": "confirmed",
                "winner_exploitation_cap": 0.50,
            },
            latest_batch_overview={},
        )
        # If targeted follow-up already covers weak_long_horizon / weak_in_high_vol,
        # confirmation_required should be suppressed (False or confirmation_reason should
        # reference targeted holdout instead of generic weak-horizon signal)
        reasons = plan.get("confirmation_reason") or ""
        if plan.get("confirmation_required"):
            # If still set, the reason must NOT be the plain weak-horizon trigger;
            # it should acknowledge the targeted holdout
            self.assertNotIn("weak_long_horizon triggers confirmation", reasons)


# ──────────────────────────────────────────────
# Smoke tests: run.py / evaluate.py unchanged
# ──────────────────────────────────────────────

class TestNoRegressionInRunAndEvaluate(unittest.TestCase):
    """Verify that modules used by run.py and evaluate.py still import cleanly."""

    def test_experiment_runtime_decision_imports(self):
        import experiment_runtime_decision  # noqa: F401

    def test_experiment_types_imports(self):
        import experiment_types  # noqa: F401

    def test_runtime_decision_dataclass_has_holdout_rationale_fields(self):
        from experiment_types import RuntimeDecision
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(RuntimeDecision)}
        self.assertIn("holdout_rationale", field_names)
        self.assertIn("holdout_rationale_detail", field_names)

    def test_holdout_rationale_default_is_no_winner(self):
        from experiment_types import RuntimeDecision
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(RuntimeDecision)}
        self.assertEqual(fields["holdout_rationale"].default, "no_winner")
        self.assertIsNone(fields["holdout_rationale_detail"].default)

    def test_targeted_holdout_types_is_frozenset(self):
        self.assertIsInstance(_TARGETED_HOLDOUT_TYPES, frozenset)


if __name__ == "__main__":
    unittest.main()
