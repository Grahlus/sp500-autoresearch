"""ResearchPlan dataclass and validation.

A ResearchPlan is the output of the Layer 2 planner. It is advisory:
the executor (Layer 1) consumes it as a high-level policy hint but never
bypasses its own safety/guardrail logic because of it.

Validation rules
----------------
- plan_id: non-empty string
- created_at, valid_until: ISO-8601 datetime strings, valid_until > created_at
- family_budgets: dict {family -> 0..1}, values sum <= 1.0, no negative values
- template_budgets: dict {template_id -> 0..1}, no negative values
- new_idea_quota, uncommon_idea_quota: None or 0..1
- repeat_branch_cap: None or >= 1
- structural_execution_lane_enabled: bool
- structural_execution_lane_budget: None or >= 0
- structural_execution_lane_min_candidates: None or >= 1
- structural_execution_lane_repeat_cap: None or >= 1
- cycle_mode_override: None or one of the allowed cycle modes
- families_paused: list of known family names (unknown names are dropped silently)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

ALLOWED_CYCLE_MODES: frozenset[str] = frozenset(
    {"exploration", "confirmation", "holdout", "diagnostics", "stagnation_escape", "large_search"}
)

ALLOWED_FAMILIES: frozenset[str] = frozenset(
    {"momentum", "superstock", "ml_ranker", "rl_bandit"}
)


@dataclass
class ResearchPlan:
    """High-level planning directive produced by the Layer 2 planner.

    Fields that the executor applies as policy hints (not hard overrides):

    family_budgets
        Per-family fractional allocation hint.  Keys are family names; values
        are floats in [0, 1] with sum <= 1.  The executor uses these as
        ``per_family_budgets`` seed for ``build_proposal_request()``.

    template_budgets
        Relative weight hints for structural templates.  Keys are template IDs;
        values in [0, 1].  Higher weight → more slots allowed per cycle.

    holdout_priorities
        Config hashes to prioritise for holdout validation this planning window.

    new_idea_quota / uncommon_idea_quota
        Fraction of budget to allocate to new / uncommon ideas.

    repeat_branch_cap
        Maximum branch-repeat slots per cycle.

    structural_execution_lane_enabled / structural_execution_lane_budget
        Protected lane for structurally novel ideas.  The executor still caps
        this deterministically and can reduce it under hard safety/resource
        constraints.

    stagnation_escape_active
        If True, executor should widen exploration even if not yet hitting the
        internal stagnation threshold.

    ml_probe_allowed / rl_probe_allowed
        Allow ML/RL families to be included in multi-family batches this window.

    cycle_mode_override
        If set, the executor MAY use this as a ``cycle_mode`` hint.  The
        runtime-decision guardrail can still override it.

    families_paused
        Families the planner recommends skipping this window.  The executor
        may ignore this if confirmation or holdout logic requires them.

    branches_deprioritized
        Branch root config hashes the planner recommends spending fewer slots on.
    """

    plan_id: str
    created_at: str          # ISO-8601 UTC
    valid_until: str         # ISO-8601 UTC
    planner_name: str
    planner_model: str
    planner_provider: str
    planning_window_hours: float
    source_report_ids: list[str]
    source_dashboard_timestamp: str | None

    # Core planning parameters
    family_budgets: dict[str, float]
    template_budgets: dict[str, float]
    holdout_priorities: list[str]
    new_idea_quota: float | None
    uncommon_idea_quota: float | None
    repeat_branch_cap: int | None
    structural_execution_lane_enabled: bool
    structural_execution_lane_budget: int | None
    structural_execution_lane_min_candidates: int | None
    structural_execution_lane_repeat_cap: int | None
    stagnation_escape_active: bool
    ml_probe_allowed: bool
    rl_probe_allowed: bool
    rationale: str

    # Optional overrides
    cycle_mode_override: str | None = None
    targeted_validation_priority: str | None = None
    families_paused: list[str] = field(default_factory=list)
    branches_deprioritized: list[str] = field(default_factory=list)
    source_idea_ids: list[str] = field(default_factory=list)


@dataclass
class PlanLoadResult:
    """Result of attempting to load a plan from disk."""

    plan: ResearchPlan | None
    loaded: bool
    reason: str    # "ok", "no_file", "expired", "malformed", "validation_error", "fallback"
    plan_id: str | None
    valid_until: str | None


def _parse_dt(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse datetime: {value!r}") from exc


def validate_plan(data: Any) -> tuple[ResearchPlan | None, str]:
    """Parse and validate a dict into a ResearchPlan.

    Returns (plan, "ok") on success or (None, reason_string) on failure.
    Reason strings are human-readable and safe to log.
    """
    if not isinstance(data, dict):
        return None, "not_a_dict"

    # Required string fields
    for key in ("plan_id", "created_at", "valid_until", "planner_name",
                "planner_model", "planner_provider", "rationale"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            return None, f"missing_or_empty:{key}"

    plan_id = str(data["plan_id"]).strip()
    if not plan_id:
        return None, "empty_plan_id"

    # Datetime fields
    try:
        created_at_dt = _parse_dt(data["created_at"])
        valid_until_dt = _parse_dt(data["valid_until"])
    except ValueError as exc:
        return None, f"invalid_datetime:{exc}"

    if valid_until_dt <= created_at_dt:
        return None, "valid_until_not_after_created_at"

    # Numeric fields
    planning_window_hours = float(data.get("planning_window_hours") or 6.0)
    if planning_window_hours <= 0:
        return None, "planning_window_hours_must_be_positive"

    # family_budgets
    family_budgets_raw = data.get("family_budgets") or {}
    if not isinstance(family_budgets_raw, dict):
        return None, "family_budgets_must_be_dict"
    family_budgets: dict[str, float] = {}
    for k, v in family_budgets_raw.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, f"family_budgets_bad_value:{k}"
        if fv < 0:
            return None, f"family_budgets_negative:{k}"
        family_budgets[str(k)] = fv
    total_fb = sum(family_budgets.values())
    if total_fb > 1.001:
        return None, f"family_budgets_sum_exceeds_1:{total_fb:.4f}"

    # template_budgets
    template_budgets_raw = data.get("template_budgets") or {}
    if not isinstance(template_budgets_raw, dict):
        return None, "template_budgets_must_be_dict"
    template_budgets: dict[str, float] = {}
    for k, v in template_budgets_raw.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None, f"template_budgets_bad_value:{k}"
        if fv < 0:
            return None, f"template_budgets_negative:{k}"
        template_budgets[str(k)] = fv

    # Quota fields
    def _opt_frac(key: str) -> tuple[float | None, str | None]:
        raw = data.get(key)
        if raw is None:
            return None, None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, f"{key}_not_numeric"
        if v < 0 or v > 1:
            return None, f"{key}_out_of_range:{v}"
        return v, None

    new_idea_quota, err = _opt_frac("new_idea_quota")
    if err:
        return None, err
    uncommon_idea_quota, err = _opt_frac("uncommon_idea_quota")
    if err:
        return None, err

    # Boolean fields
    def _bool(key: str, default: bool = False) -> bool:
        v = data.get(key)
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        return str(v).lower() in {"true", "1", "yes"}

    # repeat_branch_cap
    rbc_raw = data.get("repeat_branch_cap")
    repeat_branch_cap: int | None = None
    if rbc_raw is not None:
        try:
            repeat_branch_cap = int(rbc_raw)
        except (TypeError, ValueError):
            return None, "repeat_branch_cap_not_integer"
        if repeat_branch_cap < 1:
            return None, f"repeat_branch_cap_must_be_ge_1:{repeat_branch_cap}"

    structural_execution_lane_enabled = _bool("structural_execution_lane_enabled", True)

    structural_lane_budget_raw = data.get("structural_execution_lane_budget")
    structural_execution_lane_budget: int | None = None
    if structural_lane_budget_raw is not None:
        try:
            structural_execution_lane_budget = int(structural_lane_budget_raw)
        except (TypeError, ValueError):
            return None, "structural_execution_lane_budget_not_integer"
        if structural_execution_lane_budget < 0:
            return None, "structural_execution_lane_budget_must_be_ge_0"

    structural_lane_min_candidates_raw = data.get("structural_execution_lane_min_candidates")
    structural_execution_lane_min_candidates: int | None = None
    if structural_lane_min_candidates_raw is not None:
        try:
            structural_execution_lane_min_candidates = int(structural_lane_min_candidates_raw)
        except (TypeError, ValueError):
            return None, "structural_execution_lane_min_candidates_not_integer"
        if structural_execution_lane_min_candidates < 1:
            return None, "structural_execution_lane_min_candidates_must_be_ge_1"

    structural_lane_repeat_cap_raw = data.get("structural_execution_lane_repeat_cap")
    structural_execution_lane_repeat_cap: int | None = None
    if structural_lane_repeat_cap_raw is not None:
        try:
            structural_execution_lane_repeat_cap = int(structural_lane_repeat_cap_raw)
        except (TypeError, ValueError):
            return None, "structural_execution_lane_repeat_cap_not_integer"
        if structural_execution_lane_repeat_cap < 1:
            return None, "structural_execution_lane_repeat_cap_must_be_ge_1"

    stagnation_escape_active = _bool("stagnation_escape_active")
    ml_probe_allowed = _bool("ml_probe_allowed")
    rl_probe_allowed = _bool("rl_probe_allowed")

    # cycle_mode_override
    cmo_raw = data.get("cycle_mode_override")
    cycle_mode_override: str | None = None
    if cmo_raw is not None:
        cmo = str(cmo_raw).strip().lower()
        if cmo and cmo not in ALLOWED_CYCLE_MODES:
            return None, f"cycle_mode_override_unknown:{cmo}"
        cycle_mode_override = cmo or None

    # List fields
    holdout_priorities = [str(x) for x in (data.get("holdout_priorities") or [])]
    source_report_ids = [str(x) for x in (data.get("source_report_ids") or [])]
    source_idea_ids = [str(x) for x in (data.get("source_idea_ids") or [])]
    families_paused = [str(x) for x in (data.get("families_paused") or [])]
    branches_deprioritized = [str(x) for x in (data.get("branches_deprioritized") or [])]

    plan = ResearchPlan(
        plan_id=plan_id,
        created_at=data["created_at"],
        valid_until=data["valid_until"],
        planner_name=str(data["planner_name"]),
        planner_model=str(data["planner_model"]),
        planner_provider=str(data["planner_provider"]),
        planning_window_hours=planning_window_hours,
        source_report_ids=source_report_ids,
        source_dashboard_timestamp=str(data["source_dashboard_timestamp"])
        if data.get("source_dashboard_timestamp") else None,
        family_budgets=family_budgets,
        template_budgets=template_budgets,
        holdout_priorities=holdout_priorities,
        new_idea_quota=new_idea_quota,
        uncommon_idea_quota=uncommon_idea_quota,
        repeat_branch_cap=repeat_branch_cap,
        structural_execution_lane_enabled=structural_execution_lane_enabled,
        structural_execution_lane_budget=structural_execution_lane_budget,
        structural_execution_lane_min_candidates=structural_execution_lane_min_candidates,
        structural_execution_lane_repeat_cap=structural_execution_lane_repeat_cap,
        stagnation_escape_active=stagnation_escape_active,
        ml_probe_allowed=ml_probe_allowed,
        rl_probe_allowed=rl_probe_allowed,
        rationale=str(data["rationale"]),
        cycle_mode_override=cycle_mode_override,
        targeted_validation_priority=str(data["targeted_validation_priority"])
        if data.get("targeted_validation_priority") else None,
        families_paused=families_paused,
        branches_deprioritized=branches_deprioritized,
        source_idea_ids=source_idea_ids,
    )
    return plan, "ok"


def is_plan_fresh(plan: ResearchPlan) -> bool:
    """Return True if the plan's valid_until is in the future."""
    try:
        valid_until_dt = _parse_dt(plan.valid_until)
        return datetime.now(UTC) < valid_until_dt
    except ValueError:
        return False


def plan_to_dict(plan: ResearchPlan) -> dict[str, Any]:
    """Serialize a ResearchPlan to a JSON-safe dict."""
    return {
        "plan_id": plan.plan_id,
        "created_at": plan.created_at,
        "valid_until": plan.valid_until,
        "planner_name": plan.planner_name,
        "planner_model": plan.planner_model,
        "planner_provider": plan.planner_provider,
        "planning_window_hours": plan.planning_window_hours,
        "source_report_ids": list(plan.source_report_ids),
        "source_dashboard_timestamp": plan.source_dashboard_timestamp,
        "family_budgets": dict(plan.family_budgets),
        "template_budgets": dict(plan.template_budgets),
        "holdout_priorities": list(plan.holdout_priorities),
        "new_idea_quota": plan.new_idea_quota,
        "uncommon_idea_quota": plan.uncommon_idea_quota,
        "repeat_branch_cap": plan.repeat_branch_cap,
        "structural_execution_lane_enabled": plan.structural_execution_lane_enabled,
        "structural_execution_lane_budget": plan.structural_execution_lane_budget,
        "structural_execution_lane_min_candidates": plan.structural_execution_lane_min_candidates,
        "structural_execution_lane_repeat_cap": plan.structural_execution_lane_repeat_cap,
        "stagnation_escape_active": plan.stagnation_escape_active,
        "ml_probe_allowed": plan.ml_probe_allowed,
        "rl_probe_allowed": plan.rl_probe_allowed,
        "rationale": plan.rationale,
        "cycle_mode_override": plan.cycle_mode_override,
        "targeted_validation_priority": plan.targeted_validation_priority,
        "families_paused": list(plan.families_paused),
        "branches_deprioritized": list(plan.branches_deprioritized),
        "source_idea_ids": list(plan.source_idea_ids),
    }
