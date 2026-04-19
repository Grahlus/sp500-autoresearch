"""Plan persistence — load, save, and archive ResearchPlan files.

Plans live at  plans/current_plan.json
Archives live at plans/archive/<plan_id>.json

All writes are atomic (write to tmp + os.replace).
All reads degrade gracefully: missing file → PlanLoadResult(loaded=False).
"""
from __future__ import annotations

import json
import os
import pathlib
from datetime import UTC, datetime

from planner.plan_schema import (
    PlanLoadResult,
    ResearchPlan,
    is_plan_fresh,
    plan_to_dict,
    validate_plan,
)

_PLANS_DIR = pathlib.Path("plans")
_CURRENT_PLAN_PATH = _PLANS_DIR / "current_plan.json"
_ARCHIVE_DIR = _PLANS_DIR / "archive"


def _ensure_dirs() -> None:
    _PLANS_DIR.mkdir(parents=True, exist_ok=True)
    _ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def load_current_plan() -> PlanLoadResult:
    """Load and validate the current plan file.

    Returns a PlanLoadResult with reason one of:
      "ok"               — plan loaded and still fresh
      "no_file"          — plans/current_plan.json does not exist
      "expired"          — plan loaded but valid_until is in the past
      "malformed"        — file is not valid JSON
      "validation_error" — JSON parsed but failed schema validation
    """
    if not _CURRENT_PLAN_PATH.exists():
        return PlanLoadResult(plan=None, loaded=False, reason="no_file",
                              plan_id=None, valid_until=None)

    try:
        raw = _CURRENT_PLAN_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return PlanLoadResult(plan=None, loaded=False, reason=f"malformed:{exc}",
                              plan_id=None, valid_until=None)

    plan, reason = validate_plan(data)
    if plan is None:
        return PlanLoadResult(plan=None, loaded=False, reason=f"validation_error:{reason}",
                              plan_id=data.get("plan_id"), valid_until=data.get("valid_until"))

    if not is_plan_fresh(plan):
        return PlanLoadResult(plan=plan, loaded=False, reason="expired",
                              plan_id=plan.plan_id, valid_until=plan.valid_until)

    return PlanLoadResult(plan=plan, loaded=True, reason="ok",
                          plan_id=plan.plan_id, valid_until=plan.valid_until)


def save_plan(plan: ResearchPlan) -> None:
    """Atomically write plan to plans/current_plan.json."""
    _ensure_dirs()
    data = plan_to_dict(plan)
    tmp = _CURRENT_PLAN_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, _CURRENT_PLAN_PATH)


def archive_plan(plan: ResearchPlan) -> pathlib.Path:
    """Copy plan to plans/archive/<plan_id>.json.  Returns the archive path."""
    _ensure_dirs()
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in plan.plan_id)
    dest = _ARCHIVE_DIR / f"{safe_id}.json"
    data = plan_to_dict(plan)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def load_or_fallback(fallback_reason: str = "fallback") -> PlanLoadResult:
    """Load current plan; if unavailable return a typed fallback result."""
    result = load_current_plan()
    if not result.loaded:
        if result.reason == "no_file":
            return result
        return PlanLoadResult(plan=result.plan, loaded=False,
                              reason=fallback_reason if result.reason == "no_file" else result.reason,
                              plan_id=result.plan_id, valid_until=result.valid_until)
    return result
