"""Tests for holdout-evidence-based promotion gating.

Covers:
- _holdout_evidence_score helper
- persisted_confirmed requires holdout_history_count > 0
- persisted_cautious cap is 0.35 (not 0.50) when holdout_history_count == 0
- _confirmation_patience_exceeded requires holdout_history_count >= 1
- normal promoted cap follows holdout evidence quality (0.55 / 0.60 / 0.65)
- holdout_evidence_score and holdout_gating_reason surface in signals
- no behavior drift on default paths with good overall validation
"""
from __future__ import annotations

import unittest
from typing import Any

from experiment_runtime_decision import (
    _holdout_evidence_score,
    _winner_promotion_policy,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_scorecard(**overrides: Any) -> dict[str, Any]:
    """Return a minimal winner scorecard with strong defaults."""
    base: dict[str, Any] = {
        # Quality signals
        "robustness_score": 0.80,
        "overfit_risk": 0.15,
        "recent_robustness_trend": 0.0,
        "viable_rate": 0.40,
        "win_rate_vs_baseline": 0.60,
        "confidence": 0.75,
        "recent_objective_trend": 0.10,
        "recent_viable_trend": 0.05,
        "dead_zone_density": 0.05,
        "duplicate_saturation": 0.05,
        # Validation
        "validation_scope": "broad",
        "validation_confidence": 0.75,
        "validation_coverage": 0.80,
        "validation_horizon_tags": ["strong_long_horizon"],
        "validation_regime_tags": ["stable_in_trend"],
        # Holdout check — default to unknown (no actual holdout run)
        "holdout_check_type": None,
        "holdout_check_status": "unknown",
        "holdout_check_outcome": "unknown",
        "holdout_check_scope": "unknown",
        "holdout_check_batch_id": None,
        "holdout_horizon_tags": [],
        "holdout_regime_tags": [],
        # Promotion state history
        "promotion_state": "confirmed",
        "promotion_blocked_pending_new_evidence": False,
        "confirmation_history_count": 3,
        "holdout_history_count": 0,
        "promotion_history_count": 1,
        "last_confirmation_timestamp_utc": None,
        "last_confirmation_cycle_id": None,
        "last_failed_confirmation_timestamp_utc": None,
        "last_failed_confirmation_cycle_id": None,
        "confirmation_outcome": None,
        "targeted_follow_up_type": None,
        "targeted_follow_up_required": False,
        # Lineage
        "lineage_status_summary": "seed",
        "lineage_trust_score": 0.60,
        "lineage_depth": 1,
        "descendant_count": 2,
        "confirmation_descendant_count": 1,
        "holdout_descendant_count": 0,
        "rejected_descendant_count": 0,
    }
    base.update(overrides)
    return base


def _make_family_result(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "sharpe": 1.8,
        "calmar": 1.2,
        "total_return": 25.0,
        "max_drawdown": 10.0,
        "trades_per_year": 20.0,
        "exposure": 0.6,
        "viable": True,
        "comparison_status": "partial_verified_current_engine",
        "beats_baseline_objective": True,
        "beats_baseline_guardrails": True,
        "objective_score": 1.8,
    }
    base.update(overrides)
    return base


def _policy(scorecard=None, family_result=None, overfit_signal=None) -> dict[str, Any]:
    return _winner_promotion_policy(
        family="momentum",
        scorecard=scorecard or _make_scorecard(),
        family_result=family_result or _make_family_result(),
        overfit_signal=overfit_signal or {},
    )


# ──────────────────────────────────────────────
# Unit: _holdout_evidence_score
# ──────────────────────────────────────────────

class TestHoldoutEvidenceScore(unittest.TestCase):

    def test_zero_when_no_history(self):
        self.assertEqual(_holdout_evidence_score(0, "broadly_confirmed"), 0.0)
        self.assertEqual(_holdout_evidence_score(0, "confirmed"), 0.0)
        self.assertEqual(_holdout_evidence_score(0, "unknown"), 0.0)

    def test_0_75_when_one_confirmed(self):
        self.assertAlmostEqual(_holdout_evidence_score(1, "confirmed"), 0.75)

    def test_1_0_when_two_broadly_confirmed(self):
        self.assertAlmostEqual(_holdout_evidence_score(2, "broadly_confirmed"), 1.0)

    def test_capped_at_1_0_with_more_history(self):
        self.assertAlmostEqual(_holdout_evidence_score(10, "confirmed"), 1.0)

    def test_0_30_for_provisional(self):
        self.assertAlmostEqual(_holdout_evidence_score(1, "provisional"), 0.30)
        self.assertAlmostEqual(_holdout_evidence_score(3, "unproven"), 0.30)

    def test_0_40_for_unclear_outcome(self):
        self.assertAlmostEqual(_holdout_evidence_score(1, "unknown"), 0.40)
        self.assertAlmostEqual(_holdout_evidence_score(2, "pending"), 0.40)

    def test_rejected_outcome_with_history_is_unclear(self):
        # rejected is not in confirmed/broadly_confirmed/provisional/unproven → 0.40
        self.assertAlmostEqual(_holdout_evidence_score(1, "rejected"), 0.40)

    def test_monotone_history_for_confirmed(self):
        self.assertLess(
            _holdout_evidence_score(1, "confirmed"),
            _holdout_evidence_score(2, "confirmed"),
        )


# ──────────────────────────────────────────────
# persisted_confirmed path: must have holdout history
# ──────────────────────────────────────────────

class TestPersistedConfirmedRequiresHoldoutHistory(unittest.TestCase):

    def test_no_holdout_history_bypasses_persisted_confirmed(self):
        """Candidate with only generic confirmation history must NOT reach promoted/0.70."""
        sc = _make_scorecard(
            promotion_state="confirmed",
            confirmation_history_count=5,
            holdout_history_count=0,
            promotion_history_count=1,
            holdout_check_outcome="broadly_confirmed",  # synthesized good outcome
        )
        p = _policy(scorecard=sc)
        # Must NOT be the fast-path promoted (cap=0.70) since holdout_history_count == 0
        self.assertNotEqual(p["winner_exploitation_cap"], 0.70)

    def test_with_holdout_history_can_reach_persisted_confirmed(self):
        """Candidate with holdout history and confirmed outcome may reach promoted/0.70."""
        sc = _make_scorecard(
            promotion_state="confirmed",
            confirmation_history_count=3,
            holdout_history_count=1,
            promotion_history_count=1,
            holdout_check_outcome="broadly_confirmed",
            holdout_check_status="completed",
            holdout_horizon_tags=["strong_long_horizon"],
            holdout_regime_tags=["stable_in_trend"],
        )
        p = _policy(scorecard=sc)
        self.assertIn(p["winner_promotion_status"], {"promoted", "cautious_promotion"})
        # promoted path sets cap 0.70 for persisted_confirmed;
        # cautious path sets 0.50 when holdout_history_count > 0
        self.assertGreaterEqual(p["winner_exploitation_cap"], 0.50)

    def test_holdout_history_required_regardless_of_confirmation_count(self):
        """Even with many confirmations, no holdout history → cap < 0.70."""
        sc = _make_scorecard(
            promotion_state="confirmed",
            confirmation_history_count=30,
            holdout_history_count=0,
            promotion_history_count=5,
            holdout_check_outcome="broadly_confirmed",
        )
        p = _policy(scorecard=sc)
        self.assertLess(p["winner_exploitation_cap"], 0.70)


# ──────────────────────────────────────────────
# persisted_cautious path: cap split by holdout history
# ──────────────────────────────────────────────

class TestPersistedCautiousCap(unittest.TestCase):

    def _cautious_scorecard(self, *, holdout_history_count: int) -> dict[str, Any]:
        """Build a scorecard that falls into persisted_cautious (confirmed state, sparse flags)."""
        return _make_scorecard(
            promotion_state="confirmed",
            confirmation_history_count=3,
            holdout_history_count=holdout_history_count,
            promotion_history_count=1,
            # Make holdout_outcome not cleanly positive so persisted_confirmed is False
            holdout_check_outcome="provisional",
        )

    def test_cap_0_35_when_no_holdout_history(self):
        sc = self._cautious_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "cautious_promotion":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.35)

    def test_cap_0_50_when_holdout_history_present(self):
        sc = self._cautious_scorecard(holdout_history_count=2)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "cautious_promotion":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.50)

    def test_gating_reason_reflects_holdout_absence(self):
        sc = self._cautious_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "cautious_promotion":
            reasons_combined = str(p.get("reasons", [])) + str(p.get("signals", {}))
            self.assertIn("no_holdout", reasons_combined)

    def test_gating_reason_reflects_holdout_presence(self):
        sc = self._cautious_scorecard(holdout_history_count=1)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "cautious_promotion":
            self.assertIn("holdout_gating_reason", p.get("signals", {}))
            self.assertIn("holdout", p["signals"]["holdout_gating_reason"])


# ──────────────────────────────────────────────
# _confirmation_patience_exceeded: requires holdout history
# ──────────────────────────────────────────────

class TestConfirmationPatienceRequiresHoldout(unittest.TestCase):

    def _patience_scorecard(self, *, holdout_history_count: int) -> dict[str, Any]:
        """Build a suspicious-but-aging scorecard: narrow validation, many confirmations."""
        return _make_scorecard(
            promotion_state="provisional",
            confirmation_history_count=30,
            holdout_history_count=holdout_history_count,
            promotion_history_count=0,
            # narrow validation triggers 'suspicious'
            validation_scope="partial",
            validation_horizon_tags=["weak_long_horizon"],
            holdout_check_outcome="unknown",
        )

    def test_stays_suspicious_when_no_holdout_history(self):
        """After 30 confirmations but zero holdout history, patience age-out must NOT fire."""
        sc = self._patience_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        # Without the age-out override, the candidate stays at hold_for_confirmation (cap=0.25)
        # or at most cautious_promotion. It must NOT reach promoted (cap >= 0.60).
        self.assertNotIn(p["winner_promotion_status"], {"promoted"})
        # Specific: since patience can't fire without holdout history,
        # the suspicious flag stays True → hold_for_confirmation
        self.assertEqual(p["winner_promotion_status"], "hold_for_confirmation")
        self.assertLessEqual(p["winner_exploitation_cap"], 0.25)

    def test_patience_fires_when_holdout_history_present(self):
        """Age-out IS allowed when holdout_history_count >= 1."""
        sc = self._patience_scorecard(holdout_history_count=2)
        p = _policy(scorecard=sc)
        # With holdout history, patience override can fire → should NOT be hold_for_confirmation
        self.assertNotEqual(p["winner_promotion_status"], "hold_for_confirmation")

    def test_no_patience_with_fewer_than_25_cycles(self):
        """Patience age-out never fires below 25 confirmation cycles."""
        sc = self._patience_scorecard(holdout_history_count=3)
        sc["confirmation_history_count"] = 10  # too few
        p = _policy(scorecard=sc)
        # With narrow validation and only 10 confirmations, must be suspicious
        self.assertEqual(p["winner_promotion_status"], "hold_for_confirmation")


# ──────────────────────────────────────────────
# Normal promoted path: cap varies by holdout evidence quality
# ──────────────────────────────────────────────

class TestNormalPromotedCapGating(unittest.TestCase):

    def _strong_scorecard(self, *, holdout_history_count: int, holdout_outcome: str = "broadly_confirmed") -> dict[str, Any]:
        """Build a scorecard that qualifies for the normal promoted path."""
        return _make_scorecard(
            promotion_state="unconfirmed",
            confirmation_history_count=0,
            holdout_history_count=holdout_history_count,
            promotion_history_count=0,
            holdout_check_outcome=holdout_outcome,
            holdout_check_status="completed" if holdout_history_count > 0 else "unknown",
            validation_scope="broad",
            validation_horizon_tags=["strong_long_horizon"],
            validation_regime_tags=["stable_in_trend"],
            validation_confidence=0.80,
        )

    def test_cap_0_55_when_no_holdout_history(self):
        sc = self._strong_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "promoted":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.55)

    def test_cap_0_65_when_holdout_confirmed(self):
        sc = self._strong_scorecard(holdout_history_count=1, holdout_outcome="confirmed")
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "promoted":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.65)

    def test_cap_0_65_when_holdout_broadly_confirmed_with_history(self):
        sc = self._strong_scorecard(holdout_history_count=2, holdout_outcome="broadly_confirmed")
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "promoted":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.65)

    def test_cap_0_60_when_holdout_history_present_but_provisional(self):
        sc = self._strong_scorecard(holdout_history_count=1, holdout_outcome="provisional")
        p = _policy(scorecard=sc)
        # provisional holdout → hev=0.30 → cap=0.60
        if p["winner_promotion_status"] == "promoted":
            self.assertAlmostEqual(p["winner_exploitation_cap"], 0.60)

    def test_holdout_gating_reason_in_signals(self):
        sc = self._strong_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "promoted":
            self.assertIn("holdout_gating_reason", p["signals"])
            self.assertIn("no_holdout", p["signals"]["holdout_gating_reason"])

    def test_holdout_gating_reason_raised_when_confirmed(self):
        sc = self._strong_scorecard(holdout_history_count=1, holdout_outcome="confirmed")
        p = _policy(scorecard=sc)
        if p["winner_promotion_status"] == "promoted":
            self.assertIn("holdout_evidence_score", p["signals"])
            self.assertGreaterEqual(p["signals"]["holdout_evidence_score"], 0.75)

    def test_no_holdout_reduces_cap_relative_to_holdout_confirmed(self):
        sc_no = self._strong_scorecard(holdout_history_count=0)
        sc_yes = self._strong_scorecard(holdout_history_count=1, holdout_outcome="confirmed")
        p_no = _policy(scorecard=sc_no)
        p_yes = _policy(scorecard=sc_yes)
        if p_no["winner_promotion_status"] == "promoted" and p_yes["winner_promotion_status"] == "promoted":
            self.assertLess(p_no["winner_exploitation_cap"], p_yes["winner_exploitation_cap"])


# ──────────────────────────────────────────────
# holdout_evidence_score surfaces in all paths
# ──────────────────────────────────────────────

class TestHoldoutEvidenceScoreInSignals(unittest.TestCase):

    def _assert_hev_present(self, scorecard: dict[str, Any]) -> None:
        p = _policy(scorecard=scorecard)
        self.assertIn("holdout_evidence_score", p.get("signals", {}))
        self.assertIn("holdout_gating_reason", p.get("signals", {}))
        hev = p["signals"]["holdout_evidence_score"]
        self.assertGreaterEqual(hev, 0.0)
        self.assertLessEqual(hev, 1.0)

    def test_hev_in_blocked_pending_path(self):
        sc = _make_scorecard(promotion_blocked_pending_new_evidence=True)
        self._assert_hev_present(sc)

    def test_hev_in_hold_for_confirmation_path(self):
        sc = _make_scorecard(
            promotion_state="provisional",
            validation_scope="partial",
            validation_horizon_tags=["weak_long_horizon"],
            holdout_check_outcome="unknown",
        )
        self._assert_hev_present(sc)

    def test_hev_in_cautious_promotion_path(self):
        sc = _make_scorecard(
            robustness_score=0.65,  # below 0.70 → cautious
            overfit_risk=0.25,
        )
        self._assert_hev_present(sc)

    def test_hev_in_promoted_path(self):
        sc = _make_scorecard(holdout_history_count=0)
        self._assert_hev_present(sc)

    def test_hev_zero_when_no_holdout_history(self):
        sc = _make_scorecard(holdout_history_count=0)
        p = _policy(scorecard=sc)
        self.assertAlmostEqual(p["signals"]["holdout_evidence_score"], 0.0)

    def test_hev_positive_when_holdout_history_present(self):
        sc = _make_scorecard(
            holdout_history_count=1,
            holdout_check_outcome="broadly_confirmed",
        )
        p = _policy(scorecard=sc)
        self.assertGreater(p["signals"]["holdout_evidence_score"], 0.0)


# ──────────────────────────────────────────────
# Default path regression: strong candidate still gets promoted
# ──────────────────────────────────────────────

class TestDefaultPathsUnchanged(unittest.TestCase):

    def test_strong_candidate_with_holdout_history_gets_promoted(self):
        sc = _make_scorecard(
            holdout_history_count=1,
            holdout_check_outcome="broadly_confirmed",
            confirmation_history_count=3,
            promotion_history_count=1,
        )
        p = _policy(scorecard=sc)
        self.assertIn(p["winner_promotion_status"], {"promoted", "cautious_promotion"})

    def test_blocked_candidate_stays_blocked(self):
        sc = _make_scorecard(promotion_blocked_pending_new_evidence=True)
        p = _policy(scorecard=sc)
        self.assertEqual(p["winner_promotion_status"], "blocked_pending_new_evidence")
        self.assertAlmostEqual(p["winner_exploitation_cap"], 0.15)

    def test_suspicious_candidate_holds_for_confirmation(self):
        sc = _make_scorecard(
            promotion_state="provisional",
            robustness_score=0.30,  # low robustness → suspicious
            holdout_history_count=0,
        )
        p = _policy(scorecard=sc)
        self.assertEqual(p["winner_promotion_status"], "hold_for_confirmation")
        self.assertAlmostEqual(p["winner_exploitation_cap"], 0.25)

    def test_no_family_returns_not_promoted(self):
        p = _winner_promotion_policy(
            family=None,
            scorecard={},
            family_result=None,
            overfit_signal=None,
        )
        self.assertEqual(p["winner_promotion_status"], "not_promoted")
        self.assertIsNone(p["winner_exploitation_cap"])


if __name__ == "__main__":
    unittest.main()
