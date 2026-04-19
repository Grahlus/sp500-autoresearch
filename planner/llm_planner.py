"""MiniMax-backed Layer 2 planner."""
from __future__ import annotations

import json
import logging
import textwrap
import traceback
from datetime import UTC, datetime, timedelta
from typing import Any

from llm.providers import DEFAULT_MINIMAX_MODEL, build_minimax_provider
from planner.plan_schema import ResearchPlan, validate_plan

logger = logging.getLogger(__name__)

_PLANNER_MODEL = DEFAULT_MINIMAX_MODEL
_DEFAULT_TIMEOUT = 120
_DEFAULT_WINDOW_HOURS = 2.0
_PROVIDER_NAME = "minimax"
_PLANNER_NAME = "minimax_planner_v1"

_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a strategic research planner for an autonomous algorithmic-trading
    research loop. Your role is ADVISORY ONLY: the deterministic executor
    (Layer 1) retains all authority over experiment execution, promotion,
    duplicate blocking, and holdout gates. Your output is a structured JSON
    planning directive that the executor MAY use as soft hints.

    You will receive a JSON context object containing:
      - scorecards: per-family health metrics
      - best_results: top experiments so far
      - template_tracking: momentum structural template progress
      - recent_batches: last few batch summaries
      - memory: lightweight loop state snapshot
      - queued_ideas: recent unconsumed idea-agent outputs (idea_id, hypothesis, family, source)

    Produce EXACTLY ONE JSON object (no prose, no markdown fences) that
    conforms to the schema below. All fields are required unless marked
    optional. Output only the JSON object - nothing else.

    SCHEMA:
    {
      "plan_id": "<unique string, e.g. plan_<timestamp>>",
      "created_at": "<ISO-8601 UTC>",
      "valid_until": "<ISO-8601 UTC, ~2h from now>",
      "planner_name": "minimax_planner_v1",
      "planner_model": "MiniMax-M2.7",
      "planner_provider": "minimax",
      "planning_window_hours": 2.0,
      "source_report_ids": [],
      "source_idea_ids": [],
      "source_dashboard_timestamp": null,
      "family_budgets": {"momentum": 0.6, "superstock": 0.3, "ml_ranker": 0.1},
      "template_budgets": {},
      "holdout_priorities": [],
      "new_idea_quota": null,
      "uncommon_idea_quota": null,
      "repeat_branch_cap": null,
      "structural_execution_lane_enabled": true,
      "structural_execution_lane_budget": null,
      "structural_execution_lane_min_candidates": null,
      "structural_execution_lane_repeat_cap": null,
      "stagnation_escape_active": false,
      "ml_probe_allowed": false,
      "rl_probe_allowed": false,
      "rationale": "<2-3 sentence explanation of your allocations>",
      "cycle_mode_override": null,
      "targeted_validation_priority": null,
      "families_paused": [],
      "branches_deprioritized": []
    }

    CONSTRAINTS:
    - family_budgets values must sum to <= 1.0, all >= 0
    - template_budgets values must all be >= 0
    - new_idea_quota and uncommon_idea_quota must be null or 0.0-1.0
    - repeat_branch_cap must be null or integer >= 1
    - structural_execution_lane_enabled must be boolean
    - structural_execution_lane_budget must be null or integer >= 0
    - structural_execution_lane_min_candidates must be null or integer >= 1
    - structural_execution_lane_repeat_cap must be null or integer >= 1
    - cycle_mode_override must be null or one of:
      exploration, confirmation, holdout, diagnostics,
      stagnation_escape, large_search
    - source_idea_ids: list the idea_ids from queued_ideas that most influenced your plan
    - Output only valid JSON, no markdown, no comments, no trailing commas
    """
).strip()


def _build_user_prompt(context: dict[str, Any], window_hours: float) -> str:
    now = datetime.now(UTC)
    valid_until = now + timedelta(hours=window_hours)
    context_json = json.dumps(context, indent=2, default=str)
    return (
        f"Current UTC time: {now.isoformat()}\n"
        f"Planning window valid until: {valid_until.isoformat()}\n\n"
        f"Research state context:\n{context_json}\n\n"
        "Produce the planning directive JSON now."
    )


def _extract_json_text(response_text: str) -> str:
    raw_text = response_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    return raw_text


def call_minimax_planner(
    context: dict[str, Any],
    *,
    window_hours: float = _DEFAULT_WINDOW_HOURS,
    timeout: int = _DEFAULT_TIMEOUT,
    provider=None,
) -> ResearchPlan | None:
    """Call MiniMax through the Anthropic-compatible SDK path."""
    try:
        provider = provider or build_minimax_provider(model=_PLANNER_MODEL)
    except Exception:
        logger.warning("call_minimax_planner: provider setup failed\n%s", traceback.format_exc())
        return None

    user_prompt = _build_user_prompt(context, window_hours)

    try:
        result = provider.create_message(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=2048,
            temperature=0.0,
        )
    except Exception:
        logger.warning("call_minimax_planner: request failed\n%s", traceback.format_exc())
        return None

    raw_text = _extract_json_text(result.text)
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning("call_minimax_planner: JSON parse failed — %s\ntext=%s", exc, raw_text[:400])
        return None

    plan, reason = validate_plan(data)
    if plan is None:
        logger.warning("call_minimax_planner: plan validation failed — %s", reason)
        return None

    logger.info("call_minimax_planner: plan %s valid until %s", plan.plan_id, plan.valid_until)
    return plan


def build_minimax_plan(
    context: dict[str, Any],
    *,
    window_hours: float = _DEFAULT_WINDOW_HOURS,
    timeout: int = _DEFAULT_TIMEOUT,
    provider=None,
) -> ResearchPlan | None:
    """Compatibility wrapper used by the planner CLI."""
    return call_minimax_planner(
        context,
        window_hours=window_hours,
        timeout=timeout,
        provider=provider,
    )
