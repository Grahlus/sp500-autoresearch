"""Template entry and yield tracking for structural momentum templates.

Observability-only module — no strategy logic, no changes to index.csv schema.
Tracks four protected structural templates through proposal, execution, and yield.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd


TRACKED_TEMPLATES: frozenset[str] = frozenset({
    "momentum_vol_scaling_v1",
    "momentum_vol_scaling_downside_v1",
    "momentum_drift_regime_v1",
    "momentum_drift_regime_strict_v1",
})

# Mirrors experiment_refinement._STRUCTURAL_EVIDENCE_THRESHOLD
STRUCTURAL_EVIDENCE_THRESHOLD: int = 8
# Mirrors experiment_refinement._STRUCTURAL_REPEAT_CAP
STRUCTURAL_REPEAT_CAP: int = 2

# A branch that owns this share of total family descendants is "dominating"
LINEAGE_DOMINANCE_THRESHOLD: float = 0.70

# How many recent experiments to scan when checking for structural floor activity
RECENT_FLOOR_WINDOW: int = 30


@dataclass(frozen=True)
class TemplateEntryRecord:
    template_id: str
    proposal_count: int
    experiment_count: int
    last_seen_timestamp_utc: str | None


@dataclass(frozen=True)
class TemplateYieldRecord:
    template_id: str
    experiment_count: int
    viable_count: int
    viable_rate: float
    baseline_beat_count: int
    baseline_beat_rate: float
    robust_descendant_count: int
    best_objective_score: float | None
    recent_trend: float
    idea_state: str
    last_seen_timestamp_utc: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _trend(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0.0
    midpoint = len(numeric) // 2
    earlier = float(numeric.iloc[:midpoint].mean())
    later = float(numeric.iloc[midpoint:].mean())
    return later - earlier


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return value
    return value


def _robust(row: pd.Series) -> bool:
    viable = _truthy(row.get("viable"))
    beats = _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))
    score = _clean_float(row.get("objective_score")) or 0.0
    return bool(viable and (beats or score >= 0.25))


def _state_from_counts(*, experiment_count: int, viable_count: int, baseline_beat_count: int) -> str:
    if experiment_count == 0:
        return "untested"
    viable_rate = viable_count / experiment_count
    beat_rate = baseline_beat_count / experiment_count
    if viable_rate >= 0.25 and beat_rate >= 0.15:
        return "promising"
    if viable_rate > 0.0 or beat_rate > 0.0:
        return "active"
    if experiment_count >= 6:
        return "retired"
    return "active"


# ---------------------------------------------------------------------------
# Public report builders
# ---------------------------------------------------------------------------

def build_template_entry_report(
    index: pd.DataFrame,
    *,
    tracked_templates: frozenset[str] = TRACKED_TEMPLATES,
) -> dict[str, TemplateEntryRecord]:
    """Count proposals and executions per tracked template.

    Always returns a record for every template in tracked_templates, even if
    experiment_count == 0, so callers can detect templates that have never run.
    """
    records: dict[str, TemplateEntryRecord] = {}
    for tid in tracked_templates:
        if index.empty or "template_id" not in index.columns:
            records[tid] = TemplateEntryRecord(
                template_id=tid,
                proposal_count=0,
                experiment_count=0,
                last_seen_timestamp_utc=None,
            )
            continue
        mask = index["template_id"].astype(str) == tid
        sub = index[mask]
        experiment_count = len(sub)
        proposal_count = 0
        if not sub.empty and "source_proposal_id" in sub.columns:
            proposal_count = int(sub["source_proposal_id"].dropna().astype(str).nunique())
        last_seen = None
        if not sub.empty and "timestamp_utc" in sub.columns:
            ts = sub["timestamp_utc"].dropna()
            if not ts.empty:
                ordered = sub.sort_values("timestamp_utc", kind="mergesort")
                last_seen = str(ordered["timestamp_utc"].iloc[-1])
        records[tid] = TemplateEntryRecord(
            template_id=tid,
            proposal_count=proposal_count,
            experiment_count=experiment_count,
            last_seen_timestamp_utc=last_seen,
        )
    return records


def build_template_yield_report(
    index: pd.DataFrame,
    *,
    idea_yield_summary: dict[str, Any] | None = None,
    tracked_templates: frozenset[str] = TRACKED_TEMPLATES,
    family: str = "momentum",
) -> dict[str, TemplateYieldRecord]:
    """Per-template yield metrics.

    Merges raw index aggregation with richer IdeaYieldRecord data from
    idea_yield_summary["by_template_id"] when available.  Returns a record
    for every tracked template, including zero-experiment sentinels.
    """
    by_template_id: dict[str, Any] = {}
    if isinstance(idea_yield_summary, dict):
        by_template_id = dict(idea_yield_summary.get("by_template_id") or {})

    records: dict[str, TemplateYieldRecord] = {}
    for tid in tracked_templates:
        yield_key = f"{family}::template::{tid}"
        yield_rec: dict[str, Any] = by_template_id.get(yield_key) or {}

        if not index.empty and "template_id" in index.columns:
            mask = index["template_id"].astype(str) == tid
            sub = index[mask].copy()
        else:
            sub = pd.DataFrame()

        experiment_count = int(yield_rec.get("idea_attempt_count") or len(sub))
        viable_count = int(yield_rec.get("idea_viable_count") or (
            int(sub["viable"].map(_truthy).sum()) if not sub.empty and "viable" in sub.columns else 0
        ))
        baseline_beat_count = int(yield_rec.get("idea_baseline_beat_count") or (
            int((
                sub["beats_baseline_objective"].map(_truthy) | sub["beats_baseline_guardrails"].map(_truthy)
            ).sum())
            if not sub.empty and {"beats_baseline_objective", "beats_baseline_guardrails"} <= set(sub.columns)
            else 0
        ))
        robust_count = int(yield_rec.get("idea_robust_descendant_count") or (
            int(sub.apply(_robust, axis=1).sum()) if not sub.empty else 0
        ))
        best_score: float | None = None
        if not sub.empty and "objective_score" in sub.columns:
            scores = pd.to_numeric(sub["objective_score"], errors="coerce").dropna()
            if not scores.empty:
                best_score = round(float(scores.max()), 6)
        recent_trend = float(yield_rec.get("idea_recent_trend") or (
            _trend(sub.sort_values("timestamp_utc", kind="mergesort")["objective_score"])
            if not sub.empty and "objective_score" in sub.columns
            else 0.0
        ))
        idea_state = str(yield_rec.get("idea_state") or _state_from_counts(
            experiment_count=experiment_count,
            viable_count=viable_count,
            baseline_beat_count=baseline_beat_count,
        ))
        last_seen = yield_rec.get("last_seen_timestamp_utc")
        if not last_seen and not sub.empty and "timestamp_utc" in sub.columns:
            ts = sub["timestamp_utc"].dropna()
            if not ts.empty:
                last_seen = str(sub.sort_values("timestamp_utc", kind="mergesort")["timestamp_utc"].iloc[-1])

        total = max(1, experiment_count)
        records[tid] = TemplateYieldRecord(
            template_id=tid,
            experiment_count=experiment_count,
            viable_count=viable_count,
            viable_rate=round(viable_count / total, 6),
            baseline_beat_count=baseline_beat_count,
            baseline_beat_rate=round(baseline_beat_count / total, 6),
            robust_descendant_count=robust_count,
            best_objective_score=best_score,
            recent_trend=round(recent_trend, 6),
            idea_state=idea_state,
            last_seen_timestamp_utc=last_seen,
        )
    return records


def detect_lineage_dominance(
    branch_summaries: dict[str, Any],
    *,
    family: str = "momentum",
    threshold: float = LINEAGE_DOMINANCE_THRESHOLD,
) -> dict[str, Any]:
    """Check if one lineage branch is consuming an outsized share of experiments.

    Computes each branch's share of total family descendants.  A branch that
    owns >= threshold of all descendants is flagged as dominating.

    Args:
        branch_summaries: the ``lineage["branch_summaries"]`` dict from the
            lineage summary (keyed by branch_root_config_hash).
        family: which family's branches to inspect.
        threshold: fractional share above which a branch is "dominant".
    """
    family_branches: dict[str, Any] = dict(branch_summaries.get(family) or {})
    if not family_branches:
        return {
            "dominant": False,
            "dominant_branch_root": None,
            "dominant_branch_share": 0.0,
            "total_descendants": 0,
            "threshold": threshold,
            "warning": None,
        }
    total_descendants = sum(
        int(b.get("descendant_count") or 0) for b in family_branches.values()
    )
    if total_descendants == 0:
        return {
            "dominant": False,
            "dominant_branch_root": None,
            "dominant_branch_share": 0.0,
            "total_descendants": 0,
            "threshold": threshold,
            "warning": None,
        }
    top_root = max(
        family_branches,
        key=lambda h: int(family_branches[h].get("descendant_count") or 0),
    )
    top_count = int(family_branches[top_root].get("descendant_count") or 0)
    share = top_count / total_descendants
    dominant = share >= threshold
    return {
        "dominant": dominant,
        "dominant_branch_root": top_root if dominant else None,
        "dominant_branch_share": round(share, 4),
        "total_descendants": total_descendants,
        "threshold": threshold,
        "warning": (
            f"Branch {top_root} owns {share:.0%} of {family} descendants "
            f"({top_count}/{total_descendants}). "
            "Structural templates may be starved if this branch keeps dominating."
        ) if dominant else None,
    }


def detect_structural_floor_active(
    index: pd.DataFrame,
    *,
    tracked_templates: frozenset[str] = TRACKED_TEMPLATES,
    recent_window: int = RECENT_FLOOR_WINDOW,
) -> dict[str, Any]:
    """Check whether any tracked template appeared in the most recent experiments.

    Uses the last ``recent_window`` rows (sorted by timestamp_utc) to detect
    whether the structural floor is actually producing real experiments.
    """
    never_run = sorted(tid for tid in tracked_templates)
    if index.empty or "template_id" not in index.columns:
        return {
            "any_structural_in_recent": False,
            "templates_in_recent": [],
            "templates_never_run": never_run,
            "recent_window": recent_window,
        }

    all_template_ids = index["template_id"].astype(str)
    never_run = sorted(
        tid for tid in tracked_templates
        if not (all_template_ids == tid).any()
    )

    sort_cols = ["timestamp_utc"] if "timestamp_utc" in index.columns else []
    if sort_cols:
        recent = index.sort_values(sort_cols, kind="mergesort").tail(recent_window)
    else:
        recent = index.tail(recent_window)

    recent_tids = set(recent["template_id"].astype(str).unique())
    in_recent = sorted(tid for tid in tracked_templates if tid in recent_tids)

    return {
        "any_structural_in_recent": bool(in_recent),
        "templates_in_recent": in_recent,
        "templates_never_run": never_run,
        "recent_window": recent_window,
    }


def build_structural_balance_policy(
    index: pd.DataFrame,
    *,
    tracked_templates: frozenset[str] = TRACKED_TEMPLATES,
    evidence_threshold: int = STRUCTURAL_EVIDENCE_THRESHOLD,
    repeat_cap: int = STRUCTURAL_REPEAT_CAP,
) -> dict[str, Any]:
    """Describe the per-template balancing policy state.

    Returns a dict showing, for each tracked template, whether it is currently
    floor-protected (below evidence_threshold), yield-eligible (above it), and
    what the anti-monopoly repeat_cap is.  Also summarises how many templates
    are "behind" vs "ahead" so the runtime rationale can be populated.
    """
    template_exec_counts: dict[str, int] = {}
    if not index.empty and "template_id" in index.columns:
        counts = index["template_id"].astype(str).value_counts()
        template_exec_counts = {k: int(v) for k, v in counts.items()}

    per_template: dict[str, Any] = {}
    for tid in tracked_templates:
        execs = template_exec_counts.get(tid, 0)
        floor_protected = execs < evidence_threshold
        per_template[tid] = {
            "experiment_count": execs,
            "floor_protected": floor_protected,
            "yield_eligible": not floor_protected,
            "status": "behind" if floor_protected else "ahead",
        }

    behind = sorted(tid for tid, rec in per_template.items() if rec["floor_protected"])
    ahead = sorted(tid for tid, rec in per_template.items() if not rec["floor_protected"])
    all_graduated = len(behind) == 0

    return {
        "evidence_threshold": evidence_threshold,
        "repeat_cap": repeat_cap,
        "behind_count": len(behind),
        "ahead_count": len(ahead),
        "all_graduated": all_graduated,
        "templates_behind": behind,
        "templates_ahead": ahead,
        "anti_monopoly_active": bool(behind) and bool(ahead),
        "structural_budget_share_per_behind": round(1.0 / max(1, len(behind)), 3) if behind else 0.0,
        "per_template": per_template,
        "rationale": (
            f"{len(behind)} template(s) below evidence threshold ({evidence_threshold}): "
            f"{behind}. Anti-monopoly repeat_cap={repeat_cap} applied to ahead templates: {ahead}."
        ) if behind else (
            f"All {len(tracked_templates)} structural templates have reached evidence threshold "
            f"({evidence_threshold}). Yield-based allocation active."
        ),
    }


def build_full_template_tracking_report(
    index: pd.DataFrame,
    *,
    idea_yield_summary: dict[str, Any] | None = None,
    branch_summaries: dict[str, Any] | None = None,
    tracked_templates: frozenset[str] = TRACKED_TEMPLATES,
    family: str = "momentum",
    recent_window: int = RECENT_FLOOR_WINDOW,
    lineage_dominance_threshold: float = LINEAGE_DOMINANCE_THRESHOLD,
) -> dict[str, Any]:
    """Combine all four observability signals into one JSON-serialisable dict.

    Schema::

        {
          "timestamp_utc": str,
          "family": str,
          "tracked_templates": list[str],
          "template_entry":  {template_id: TemplateEntryRecord, ...},
          "template_yield":  {template_id: TemplateYieldRecord, ...},
          "lineage_dominance": {...},
          "structural_floor_active": {...},
        }
    """
    entry = build_template_entry_report(index, tracked_templates=tracked_templates)
    yield_ = build_template_yield_report(
        index,
        idea_yield_summary=idea_yield_summary,
        tracked_templates=tracked_templates,
        family=family,
    )
    dominance = detect_lineage_dominance(
        branch_summaries or {},
        family=family,
        threshold=lineage_dominance_threshold,
    )
    floor = detect_structural_floor_active(index, tracked_templates=tracked_templates, recent_window=recent_window)
    balance = build_structural_balance_policy(
        index,
        tracked_templates=tracked_templates,
        evidence_threshold=STRUCTURAL_EVIDENCE_THRESHOLD,
        repeat_cap=STRUCTURAL_REPEAT_CAP,
    )

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "family": family,
        "tracked_templates": sorted(tracked_templates),
        "template_entry": {tid: asdict(rec) for tid, rec in entry.items()},
        "template_yield": {tid: asdict(rec) for tid, rec in yield_.items()},
        "lineage_dominance": dominance,
        "structural_floor_active": floor,
        "structural_balance_policy": balance,
    }
    return _json_safe(report)
