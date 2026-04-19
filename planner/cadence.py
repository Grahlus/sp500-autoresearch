"""Deterministic helper for Layer 2 planner freshness and cadence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from planner.plan_schema import ResearchPlan, is_plan_fresh


@dataclass(frozen=True)
class PlannerCadenceDecision:
    should_replan: bool
    reason: str
    plan_age_hours: float | None


def plan_age_hours(plan: ResearchPlan | None, now: datetime | None = None) -> float | None:
    if plan is None:
        return None
    now = now or datetime.now(UTC)
    try:
        created_at = datetime.fromisoformat(plan.created_at)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age = now - created_at
        return max(0.0, age.total_seconds() / 3600.0)
    except Exception:
        return None


def should_replan(
    plan: ResearchPlan | None,
    *,
    min_replan_interval_hours: float,
    now: datetime | None = None,
) -> PlannerCadenceDecision:
    now = now or datetime.now(UTC)
    if plan is None:
        return PlannerCadenceDecision(True, "no_current_plan", None)

    age_hours = plan_age_hours(plan, now=now)
    if not is_plan_fresh(plan):
        return PlannerCadenceDecision(True, "plan_expired", age_hours)

    if age_hours is not None and age_hours < min_replan_interval_hours:
        return PlannerCadenceDecision(False, "plan_fresh_within_min_interval", age_hours)

    return PlannerCadenceDecision(True, "plan_due", age_hours)

