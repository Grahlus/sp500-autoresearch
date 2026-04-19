from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_hot_index import load_hot_experiment_index
from experiment_memory import load_research_memory, save_research_memory, update_family_memory
from experiment_store import load_results_index


IDEA_YIELD_FILENAME = "idea_yield.json"
IDEA_YIELD_MEMORY_KEY = "idea_yield_states"

_STRUCTURAL_STRATEGY_TYPES: frozenset[str] = frozenset({
    "vol_regime_scaling",
    "drift_regime_conditioning",
})


@dataclass(frozen=True)
class IdeaYieldRecord:
    idea_key: str
    family: str
    idea_kind: str
    template_id: str
    strategy_type: str
    idea_source: str
    idea_family_key: str
    idea_attempt_count: int
    idea_viable_count: int
    idea_viable_rate: float
    idea_baseline_beat_count: int
    idea_baseline_beat_rate: float
    idea_descendant_count: int
    idea_robust_descendant_count: int
    idea_dead_zone_count: int
    idea_dead_zone_rate: float
    idea_repeated_failure_count: int
    idea_repeated_failure_rate: float
    idea_recent_attempt_count: int
    idea_recent_viable_rate: float
    idea_recent_baseline_beat_rate: float
    idea_recent_trend: float
    idea_quality_score: float
    idea_decay_score: float
    idea_state: str
    search_priority: float
    confidence: float
    evidence_weight: float
    last_seen_timestamp_utc: str | None
    last_success_timestamp_utc: str | None
    last_failure_timestamp_utc: str | None
    recent_objective_mean: float | None = None
    recent_objective_trend: float = 0.0
    recent_viable_trend: float = 0.0
    family_total_experiments: int = 0
    family_viable_rate: float = 0.0
    synthesized_template_family: str | None = None
    idea_id: str | None = None
    idea_state_previous: str | None = None
    idea_state_transition_reason: str | None = None
    structural_family_state: str | None = None
    structural_family_state_previous: str | None = None
    structural_family_promotion_state: str | None = None
    structural_family_transition_reason: str | None = None
    structural_floor_protected: bool = False
    structural_yield_supported: bool = False
    graduated_template_family: bool = False


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


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
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


def _safe_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _template_group_key(row: pd.Series) -> str:
    """Return the template-group label for a single experiment row.

    Structural templates (vol-regime scaling, drift-regime conditioning) are
    grouped by their ``strategy_type`` so their yield can be tracked separately
    from the classical momentum baseline.  All other experiments fall into the
    ``"classical"`` group.
    """
    source_type = _safe_text(row.get("source_type"), "")
    strategy_type = _safe_text(row.get("strategy_type"), "")
    synthesized_family = _safe_text(row.get("synthesized_template_family") or row.get("structural_synthesis_family"), "")
    if source_type in {"structural_synthesis", "structural_extension"}:
        return synthesized_family or strategy_type or "structural_unknown"
    if source_type == "structural_novelty" or strategy_type in _STRUCTURAL_STRATEGY_TYPES:
        return strategy_type or "structural_unknown"
    return "classical"


def _aggregate_template_groups(frame: pd.DataFrame, *, family: str) -> dict[str, Any]:
    """Aggregate idea-yield metrics per (family, template_group) composite key.

    Keys in the returned dict have the form ``"{family}::{template_group}"``.
    Each value is a ``_summarize_records`` dict with additional metadata fields
    (``group_key``, ``composite_key``, ``family``, ``experiment_count``).
    """
    if frame.empty:
        return {}
    work = frame.copy()
    work["__template_group__"] = work.apply(_template_group_key, axis=1)
    result: dict[str, Any] = {}
    for raw_group, group_frame in work.groupby("__template_group__", dropna=False):
        group_key = _safe_text(raw_group, "classical")
        composite_key = f"{family}::{group_key}"
        records = _aggregate_group(
            group_frame,
            group_field="idea_kind",
            family=family,
            category_label=f"template_group_{group_key}",
        )
        group_summary = _summarize_records(records)
        group_summary["group_key"] = group_key
        group_summary["composite_key"] = composite_key
        group_summary["family"] = family
        group_summary["experiment_count"] = int(len(group_frame))
        result[composite_key] = group_summary
    return result


def _inferred_idea_kind(row: pd.Series) -> str:
    explicit = _safe_text(row.get("idea_kind"), "")
    if explicit and explicit != "unknown":
        return explicit
    source_type = _safe_text(row.get("source_type"), "")
    strategy_type = _safe_text(row.get("strategy_type"), "")
    template_id = _safe_text(row.get("template_id"), "")
    if source_type in {"confirmation_reproduction", "confirmation_check", "holdout_check"}:
        return "holdout_confirmation"
    if source_type == "local_refinement":
        return "branch_refinement"
    if source_type in {"structural_synthesis", "structural_extension"}:
        return "new_strategy_logic_concept"
    if source_type == "idea_seed":
        return "helper_idea"
    if source_type in {"template_expansion"}:
        return "template_variation"
    if source_type == "cross_family_hybrid":
        return "cross_family_hybrid"
    if source_type == "model_based":
        return "ml_architecture_probe"
    if source_type == "policy_learning":
        return "rl_policy_probe"
    if source_type in {"broad_exploration", "saturation_escape", "external_seed"}:
        return "new_idea_probe"
    if template_id and template_id not in {"unknown", "unspecified"}:
        return template_id
    if strategy_type == "ml":
        return "ml_architecture_probe"
    if strategy_type == "rl":
        return "rl_policy_probe"
    return source_type or "new_idea_probe"


def _split_recent(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    ordered = frame.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort")
    midpoint = max(1, len(ordered) // 2)
    return ordered.head(midpoint), ordered.tail(len(ordered) - midpoint)


def _row_robust(row: pd.Series) -> bool:
    viable = _truthy(row.get("viable"))
    beats = _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))
    sharpe = _clean_float(row.get("sharpe")) or 0.0
    calmar = _clean_float(row.get("calmar")) or 0.0
    trades = _clean_float(row.get("trades_per_year")) or 0.0
    exposure = _clean_float(row.get("exposure"))
    exposure = 0.5 if exposure is None else exposure
    max_drawdown = abs(_clean_float(row.get("max_drawdown")) or 0.0)
    no_trades = _safe_text(row.get("status"), "") in {"no_trades", "invalid", "error"}
    return bool(
        viable
        and (beats or sharpe >= 0.25 or calmar >= 0.5)
        and trades >= 3.0
        and exposure >= 0.10
        and max_drawdown <= 35.0
        and not no_trades
    )


def _trend(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0.0
    midpoint = len(numeric) // 2
    earlier = float(numeric.iloc[:midpoint].mean())
    later = float(numeric.iloc[midpoint:].mean())
    return later - earlier


def _state_from_scores(*, quality: float, decay: float, attempt_count: int, viable_rate: float, baseline_rate: float) -> str:
    if attempt_count <= 0:
        return "untested"
    if quality >= 0.70 and viable_rate >= 0.25 and baseline_rate >= 0.15 and decay < 0.45:
        return "promising"
    if quality >= 0.45 and decay < 0.65:
        return "active"
    if decay >= 0.80 or (attempt_count >= 6 and viable_rate <= 0.05 and baseline_rate <= 0.05):
        return "retired"
    if decay >= 0.55:
        return "stale"
    return "active"


def _structural_family_promotion_state(
    *,
    idea_state: str,
    attempt_count: int,
    viable_count: int,
    baseline_count: int,
    quality: float,
    decay: float,
    dead_zone_rate: float,
    repeated_failure_rate: float,
    recent_viable_rate: float,
    recent_baseline_rate: float,
) -> tuple[str, str, bool, bool, bool]:
    if attempt_count <= 0:
        return "floor_protected", "no execution evidence yet", True, False, False
    if idea_state == "retired" or decay >= 0.80 or (attempt_count >= 6 and viable_count <= 0 and baseline_count <= 0):
        return "retired", "high decay or repeated failures exhausted the structural family", False, False, False
    if idea_state == "stale" or decay >= 0.55 or (attempt_count >= 4 and recent_viable_rate <= 0.10 and recent_baseline_rate <= 0.05):
        return "decaying", "recent structural evidence is weakening", False, False, False
    if viable_count <= 0 and baseline_count <= 0:
        return "floor_protected", "execution exists but no viable or baseline-beating evidence has formed yet", True, False, False

    graduated = (
        idea_state in {"promising", "active"}
        and attempt_count >= 6
        and viable_count >= 2
        and baseline_count >= 2
        and quality >= 0.70
        and decay < 0.45
        and dead_zone_rate < 0.45
        and repeated_failure_rate < 0.35
        and recent_baseline_rate >= 0.10
    )
    if graduated:
        return "graduated_structural_family", "positive viable and baseline-beating evidence is durable enough to graduate", False, True, True
    if idea_state == "promising" or quality >= 0.55 or viable_count > 0 or baseline_count > 0:
        return "yield_supported", "positive structural evidence supports yield-backed allocation", False, True, False
    return "floor_protected", "structural execution is still too thin to move beyond floor protection", True, False, False


def build_family_idea_yield_guidance(family_summary: dict[str, Any]) -> dict[str, Any]:
    top_structural = list(family_summary.get("top_synthesized_template_families") or [])
    structural_record = top_structural[0] if top_structural else None
    if not isinstance(structural_record, dict):
        structural_record = {}
    structural_state = str(
        family_summary.get("structural_family_state")
        or structural_record.get("structural_family_state")
        or structural_record.get("idea_state")
        or "untested"
    ).strip().lower() or "untested"
    promotion_state = str(
        family_summary.get("structural_family_promotion_state")
        or structural_record.get("structural_family_promotion_state")
        or "floor_protected"
    ).strip().lower() or "floor_protected"
    support_state = promotion_state
    graduated_template_family = None
    if support_state == "graduated_structural_family":
        graduated_template_family = family_summary.get("graduated_template_family") or structural_record.get("synthesized_template_family")
    structural_attempt_count = int(structural_record.get("idea_attempt_count") or 0)
    structural_recent_attempt_count = int(structural_record.get("idea_recent_attempt_count") or 0)
    if support_state == "graduated_structural_family":
        lane_budget = 3
        lane_min_candidates = 2
        lane_repeat_cap = 3
    elif support_state == "yield_supported":
        lane_budget = 2
        lane_min_candidates = 1
        lane_repeat_cap = 2
    else:
        lane_budget = 1
        lane_min_candidates = 1
        lane_repeat_cap = 1
    return {
        "idea_state": str(family_summary.get("idea_state") or "untested"),
        "idea_quality_score": round(float(family_summary.get("idea_quality_score") or 0.0), 6),
        "idea_retired_share": round(float(family_summary.get("idea_retired_share") or 0.0), 6),
        "idea_promising_share": round(float(family_summary.get("idea_promising_share") or 0.0), 6),
        "top_idea_kinds": list(family_summary.get("top_idea_kinds") or []),
        "retiring_idea_kinds": [
            record.get("idea_kind")
            for record in top_structural
            if str(record.get("idea_state") or "") in {"retired", "stale"} and record.get("idea_kind")
        ],
        "structural_family_state": structural_state,
        "structural_family_state_previous": family_summary.get("structural_family_state_previous")
        or structural_record.get("structural_family_state_previous"),
        "structural_family_promotion_state": promotion_state,
        "structural_family_transition_reason": family_summary.get("structural_family_transition_reason")
        or structural_record.get("structural_family_transition_reason"),
        "structural_family_state_reason": family_summary.get("structural_family_state_reason")
        or structural_record.get("idea_state_transition_reason"),
        "structural_family_key": family_summary.get("graduated_template_family")
        or family_summary.get("structural_family_key")
        or structural_record.get("synthesized_template_family"),
        "structural_family_quality_score": round(float(structural_record.get("idea_quality_score") or 0.0), 6),
        "structural_family_decay_score": round(float(structural_record.get("idea_decay_score") or 0.0), 6),
        "structural_family_executed_count": int(structural_record.get("idea_attempt_count") or 0),
        "structural_family_viable_count": int(structural_record.get("idea_viable_count") or 0),
        "structural_family_baseline_beat_count": int(structural_record.get("idea_baseline_beat_count") or 0),
        "structural_family_dead_zone_rate": round(float(structural_record.get("idea_dead_zone_rate") or 0.0), 6),
        "structural_family_repeated_failure_rate": round(float(structural_record.get("idea_repeated_failure_rate") or 0.0), 6),
        "structural_family_floor_protected": support_state == "floor_protected",
        "structural_family_yield_supported": support_state == "yield_supported",
        "structural_family_graduated": support_state == "graduated_structural_family",
        "graduated_template_family": graduated_template_family,
        "structural_execution_lane_enabled": True,
        "structural_execution_lane_budget": lane_budget,
        "structural_execution_lane_min_candidates": lane_min_candidates,
        "structural_execution_lane_repeat_cap": lane_repeat_cap,
        "structural_execution_lane_recent_active": bool(structural_recent_attempt_count > 0 or structural_attempt_count > 0),
        "structural_execution_lane_recent_attempt_count": structural_recent_attempt_count,
        "structural_family_feedback": {
            "old_state": family_summary.get("structural_family_state_previous")
            or structural_record.get("idea_state_previous"),
            "new_state": structural_state,
            "promotion_state": promotion_state,
            "reason": family_summary.get("structural_family_transition_reason")
            or structural_record.get("structural_family_transition_reason"),
            "graduated_template_family": graduated_template_family,
        },
    }


def _family_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "total_experiments": 0,
            "viable_rate": 0.0,
            "baseline_beat_rate": 0.0,
            "idea_yield_records": {},
            "top_idea_kinds": [],
            "top_templates": [],
            "top_strategy_types": [],
            "top_idea_sources": [],
            "top_family_idea_kinds": [],
            "idea_yield_fresh_share": 0.0,
            "idea_yield_promising_share": 0.0,
            "idea_yield_retired_share": 0.0,
        }

    return {
        "total_experiments": int(len(frame)),
        "viable_rate": float(frame["viable"].map(_truthy).mean()) if "viable" in frame else 0.0,
        "baseline_beat_rate": float(
            (
                frame["beats_baseline_objective"].map(_truthy)
                | frame["beats_baseline_guardrails"].map(_truthy)
            ).mean()
        )
        if {"beats_baseline_objective", "beats_baseline_guardrails"} <= set(frame.columns)
        else 0.0,
        "idea_yield_records": {},
        "top_idea_kinds": [],
        "top_templates": [],
        "top_strategy_types": [],
        "top_idea_sources": [],
        "top_family_idea_kinds": [],
        "idea_yield_fresh_share": 0.0,
        "idea_yield_promising_share": 0.0,
        "idea_yield_retired_share": 0.0,
    }


def _aggregate_group(
    frame: pd.DataFrame,
    *,
    group_field: str,
    family: str,
    category_label: str,
    previous_records: dict[str, dict[str, Any]] | None = None,
) -> list[IdeaYieldRecord]:
    if frame.empty:
        return []
    work = frame.copy()
    work[group_field] = work.apply(lambda row: _safe_text(row.get(group_field), "") if group_field in row else "", axis=1)
    if group_field == "idea_kind":
        work[group_field] = work.apply(_inferred_idea_kind, axis=1)
    elif group_field == "template_id":
        work[group_field] = work[group_field].map(lambda value: _safe_text(value, "unspecified"))
    elif group_field == "strategy_type":
        work[group_field] = work[group_field].map(lambda value: _safe_text(value, "classical"))
    elif group_field == "idea_source":
        work[group_field] = work[group_field].map(lambda value: _safe_text(value, "unknown"))
    else:
        work[group_field] = work[group_field].map(lambda value: _safe_text(value, "unknown"))

    records: list[IdeaYieldRecord] = []
    for raw_value, group in work.groupby(group_field, dropna=False):
        idea_value = _safe_text(raw_value, "unknown")
        if not idea_value:
            idea_value = "unknown"
        idea_kind = _inferred_idea_kind(group.iloc[0]) if group_field != "idea_kind" else idea_value
        template_id = _safe_text(group.iloc[0].get("template_id"), "unspecified")
        strategy_type = _safe_text(group.iloc[0].get("strategy_type"), "classical")
        idea_source = _safe_text(group.iloc[0].get("idea_source") or group.iloc[0].get("source_type"), "unknown")
        idea_family_key = f"{family}::{idea_kind}"
        synthesized_template_family = idea_value if group_field == "synthesized_template_family" else _safe_text(group.iloc[0].get("synthesized_template_family") or group.iloc[0].get("structural_synthesis_family"), None)
        idea_id = idea_value if group_field == "idea_id" else _safe_text(group.iloc[0].get("idea_id"), None)
        previous_record = dict((previous_records or {}).get(f"{family}::{category_label}::{idea_value}") or {})
        total = int(len(group))
        viable_mask = group["viable"].map(_truthy) if "viable" in group else pd.Series([False] * total, index=group.index)
        baseline_mask = (
            group["beats_baseline_objective"].map(_truthy) | group["beats_baseline_guardrails"].map(_truthy)
            if {"beats_baseline_objective", "beats_baseline_guardrails"} <= set(group.columns)
            else pd.Series([False] * total, index=group.index)
        )
        dead_zone_mask = (
            pd.to_numeric(group.get("dead_zone_risk", pd.Series(dtype=float)), errors="coerce").fillna(0.0) >= 0.50
        ) | group.get("status", pd.Series(dtype=str)).fillna("").astype(str).isin({"invalid", "no_trades", "error"})
        repeated_failure_mask = (~viable_mask) | pd.to_numeric(group.get("objective_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).lt(0.0)
        robust_mask = viable_mask & (baseline_mask | (pd.to_numeric(group.get("objective_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0) >= pd.to_numeric(group.get("objective_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0).median()))
        recent = group.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort").tail(max(1, min(len(group), 10))).copy()
        earlier, later = _split_recent(group)
        recent_viable_rate = float(recent["viable"].map(_truthy).mean()) if "viable" in recent else 0.0
        recent_baseline_rate = float(
            (
                recent["beats_baseline_objective"].map(_truthy)
                | recent["beats_baseline_guardrails"].map(_truthy)
            ).mean()
        ) if {"beats_baseline_objective", "beats_baseline_guardrails"} <= set(recent.columns) else 0.0
        recent_trend = _trend(recent["objective_score"]) if "objective_score" in recent else 0.0
        recent_objective_trend = _trend(group["objective_score"]) if "objective_score" in group else 0.0
        recent_viable_trend = _trend(group["viable"].map(lambda value: 1.0 if _truthy(value) else 0.0)) if "viable" in group else 0.0
        viable_count = int(viable_mask.sum())
        baseline_count = int(baseline_mask.sum())
        dead_zone_count = int(dead_zone_mask.sum())
        repeated_failure_count = int(repeated_failure_mask.sum())
        robust_count = int(robust_mask.sum())
        recent_attempt_count = int(len(recent))
        attempt_confidence = _clamp(math.log1p(total) / math.log1p(20)) if total > 0 else 0.0
        quality = _clamp(
            0.35 * _clamp(viable_count / max(1, total))
            + 0.25 * _clamp(baseline_count / max(1, total))
            + 0.15 * _clamp(robust_count / max(1, total))
            + 0.10 * _clamp((recent_viable_rate + recent_baseline_rate) / 2.0)
            + 0.10 * attempt_confidence
            + 0.05 * _clamp((recent_trend + 1.0) / 2.0)
            - 0.12 * _clamp(dead_zone_count / max(1, total))
            - 0.15 * _clamp(repeated_failure_count / max(1, total))
        )
        decay = _clamp(
            1.0
            - quality
            + 0.12 * _clamp(dead_zone_count / max(1, total))
            + 0.10 * _clamp(repeated_failure_count / max(1, total))
            - 0.05 * _clamp((recent_trend + 1.0) / 2.0)
        )
        state = _state_from_scores(
            quality=quality,
            decay=decay,
            attempt_count=total,
            viable_rate=viable_count / max(1, total),
            baseline_rate=baseline_count / max(1, total),
        )
        structural_family_promotion_state = None
        structural_floor_protected = False
        structural_yield_supported = False
        graduated_template_family = False
        structural_family_state = state if group_field == "synthesized_template_family" else None
        structural_family_transition_reason = None
        if group_field == "synthesized_template_family":
            (
                structural_family_promotion_state,
                structural_family_transition_reason,
                structural_floor_protected,
                structural_yield_supported,
                graduated_template_family,
            ) = _structural_family_promotion_state(
                idea_state=state,
                attempt_count=total,
                viable_count=viable_count,
                baseline_count=baseline_count,
                quality=quality,
                decay=decay,
                dead_zone_rate=dead_zone_count / max(1, total),
                repeated_failure_rate=repeated_failure_count / max(1, total),
                recent_viable_rate=recent_viable_rate,
                recent_baseline_rate=recent_baseline_rate,
            )
        transition_reason = structural_family_transition_reason
        if previous_record:
            previous_state = _safe_text(previous_record.get("idea_state"), "none")
            transition_reason = (
                f"{previous_state} -> {state}; {structural_family_transition_reason}"
                if structural_family_transition_reason
                else f"{previous_state} -> {state}"
            )
        search_priority = _clamp(0.35 + quality - 0.5 * decay + 0.05 * attempt_confidence)
        confidence = _clamp(attempt_confidence)
        evidence_weight = _clamp(confidence * (0.45 + 0.55 * quality))
        last_seen_timestamp = str(group.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort").iloc[-1].get("timestamp_utc") or "") or None
        success_rows = group[viable_mask]
        failure_rows = group[~viable_mask]
        last_success_timestamp = str(success_rows.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort").iloc[-1].get("timestamp_utc") or "") if not success_rows.empty else None
        last_failure_timestamp = str(failure_rows.sort_values(["timestamp_utc", "experiment_id"], kind="mergesort").iloc[-1].get("timestamp_utc") or "") if not failure_rows.empty else None
        record = IdeaYieldRecord(
            idea_key=f"{family}::{category_label}::{idea_value}",
            family=family,
            idea_kind=idea_kind,
            template_id=template_id,
            strategy_type=strategy_type,
            idea_source=idea_source,
            idea_family_key=idea_family_key,
            synthesized_template_family=synthesized_template_family,
            idea_id=idea_id,
            idea_attempt_count=total,
            idea_viable_count=viable_count,
            idea_viable_rate=round(viable_count / max(1, total), 6),
            idea_baseline_beat_count=baseline_count,
            idea_baseline_beat_rate=round(baseline_count / max(1, total), 6),
            idea_descendant_count=total,
            idea_robust_descendant_count=robust_count,
            idea_dead_zone_count=dead_zone_count,
            idea_dead_zone_rate=round(dead_zone_count / max(1, total), 6),
            idea_repeated_failure_count=repeated_failure_count,
            idea_repeated_failure_rate=round(repeated_failure_count / max(1, total), 6),
            idea_recent_attempt_count=recent_attempt_count,
            idea_recent_viable_rate=round(recent_viable_rate, 6),
            idea_recent_baseline_beat_rate=round(recent_baseline_rate, 6),
            idea_recent_trend=round(recent_trend, 6),
            idea_quality_score=round(quality, 6),
            idea_decay_score=round(decay, 6),
            idea_state=state,
            idea_state_previous=str(previous_record.get("idea_state") or "") or None,
            idea_state_transition_reason=transition_reason,
            search_priority=round(search_priority, 6),
            confidence=round(confidence, 6),
            evidence_weight=round(evidence_weight, 6),
            last_seen_timestamp_utc=last_seen_timestamp,
            last_success_timestamp_utc=last_success_timestamp,
            last_failure_timestamp_utc=last_failure_timestamp,
            recent_objective_mean=round(float(pd.to_numeric(recent.get("objective_score", pd.Series(dtype=float)), errors="coerce").mean()), 6) if "objective_score" in recent else None,
            recent_objective_trend=round(recent_objective_trend, 6),
            recent_viable_trend=round(recent_viable_trend, 6),
            family_total_experiments=total,
            family_viable_rate=round(viable_count / max(1, total), 6),
            structural_family_state=structural_family_state,
            structural_family_state_previous=str(previous_record.get("structural_family_state") or "") or None,
            structural_family_promotion_state=structural_family_promotion_state,
            structural_family_transition_reason=structural_family_transition_reason,
            structural_floor_protected=structural_floor_protected,
            structural_yield_supported=structural_yield_supported,
            graduated_template_family=graduated_template_family,
        )
        records.append(record)
    return records


def _summarize_records(records: list[IdeaYieldRecord], *, top_n: int = 5) -> dict[str, Any]:
    if not records:
        return {
            "records": {},
            "top": [],
            "idea_attempt_count": 0,
            "idea_viable_rate": 0.0,
            "idea_baseline_beat_rate": 0.0,
            "idea_descendant_count": 0,
            "idea_robust_descendant_count": 0,
            "idea_dead_zone_rate": 0.0,
            "idea_repeated_failure_rate": 0.0,
            "idea_quality_score": 0.0,
            "idea_decay_score": 1.0,
            "idea_state": "untested",
            "search_priority": 0.0,
            "confidence": 0.0,
            "evidence_weight": 0.0,
            "idea_fresh_share": 1.0,
            "idea_promising_share": 0.0,
            "idea_retired_share": 0.0,
        }
    records = sorted(records, key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))
    total = sum(record.idea_attempt_count for record in records)
    viable = sum(record.idea_viable_count for record in records)
    baseline = sum(record.idea_baseline_beat_count for record in records)
    robust = sum(record.idea_robust_descendant_count for record in records)
    dead_zone = sum(record.idea_dead_zone_count for record in records)
    failures = sum(record.idea_repeated_failure_count for record in records)
    top = [asdict(record) for record in records[:top_n]]
    state_counts: dict[str, int] = {}
    for record in records:
        state_counts[record.idea_state] = state_counts.get(record.idea_state, 0) + 1
    return {
        "records": {record.idea_key: asdict(record) for record in records},
        "top": top,
        "idea_attempt_count": total,
        "idea_viable_rate": round(viable / max(1, total), 6),
        "idea_baseline_beat_rate": round(baseline / max(1, total), 6),
        "idea_descendant_count": total,
        "idea_robust_descendant_count": robust,
        "idea_dead_zone_rate": round(dead_zone / max(1, total), 6),
        "idea_repeated_failure_rate": round(failures / max(1, total), 6),
        "idea_quality_score": round(float(sum(record.idea_quality_score for record in records) / max(1, len(records))), 6),
        "idea_decay_score": round(float(sum(record.idea_decay_score for record in records) / max(1, len(records))), 6),
        "idea_state": max(state_counts, key=state_counts.get),
        "search_priority": round(float(sum(record.search_priority for record in records) / max(1, len(records))), 6),
        "confidence": round(float(sum(record.confidence for record in records) / max(1, len(records))), 6),
        "evidence_weight": round(float(sum(record.evidence_weight for record in records) / max(1, len(records))), 6),
        "idea_fresh_share": round(sum(1 for record in records if record.idea_state == "untested") / max(1, len(records)), 6),
        "idea_promising_share": round(sum(1 for record in records if record.idea_state in {"promising", "active"}) / max(1, len(records)), 6),
        "idea_retired_share": round(sum(1 for record in records if record.idea_state == "retired") / max(1, len(records)), 6),
    }


def _has_non_empty_source_idea_ids(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, tuple):
        return len(value) > 0
    text = str(value).strip()
    return text not in {"", "[]", "()", "None", "nan"}


def build_idea_yield_summary(
    *,
    families: list[str],
    base_dir: str = "experiments",
    recent_window: int = 50,
    persist_memory: bool = True,
    use_hot_index: bool = True,
    hot_limit_per_family: int = 128,
    index: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if index is None:
        index = load_hot_experiment_index(base_dir, families=families, recent_limit_per_family=hot_limit_per_family) if use_hot_index else load_results_index(base_dir)
        if index.empty and use_hot_index:
            index = load_results_index(base_dir)
    memory = load_research_memory(base_dir)
    families = [str(family).strip().lower() for family in families if str(family).strip()]
    summary: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "families": {},
        "by_idea_kind": {},
        "by_template_id": {},
        "by_strategy_type": {},
        "by_idea_source": {},
        "by_family_idea_kind": {},
        "by_synthesized_template_family": {},
        "by_idea_id": {},
        "template_groups": {},
    }
    idea_states = memory.setdefault(IDEA_YIELD_MEMORY_KEY, {})

    if index.empty:
        for family in families:
            empty = _summarize_records([])
            summary["families"][family] = {
                "family": family,
                "recent_window": recent_window,
                **empty,
            }
        if persist_memory:
            save_research_memory(memory, base_dir)
        return summary

    for family in families:
        frame = index[index["strategy_family"].astype(str) == family].copy()
        previous_family_records = dict((idea_states.get(family) or {}))
        if frame.empty:
            family_summary = {
                "family": family,
                "recent_window": recent_window,
                **_summarize_records([]),
            }
            family_summary.update(build_family_idea_yield_guidance(family_summary))
            summary["families"][family] = family_summary
            continue
        for column in ("idea_kind", "template_id", "strategy_type", "idea_source"):
            if column not in frame.columns:
                frame[column] = None
        for column in ("synthesized_template_family", "structural_synthesis_family", "idea_id"):
            if column not in frame.columns:
                frame[column] = None
        frame["idea_kind"] = frame.apply(_inferred_idea_kind, axis=1)
        frame["template_id"] = frame["template_id"].map(lambda value: _safe_text(value, "unspecified"))
        frame["strategy_type"] = frame["strategy_type"].map(lambda value: _safe_text(value, "classical"))
        frame["idea_source"] = frame.apply(lambda row: _safe_text(row.get("idea_source") or row.get("source_type"), "unknown"), axis=1)
        frame["synthesized_template_family"] = frame.apply(
            lambda row: _safe_text(
                row.get("synthesized_template_family") or row.get("structural_synthesis_family"),
                "unknown",
            ),
            axis=1,
        )
        frame["idea_id"] = frame["idea_id"].map(lambda value: _safe_text(value, "unknown"))
        family_summary = _summarize_records([])
        grouped_sources = {
            "idea_kind": _aggregate_group(frame, group_field="idea_kind", family=family, category_label="idea_kind", previous_records=previous_family_records),
            "template_id": _aggregate_group(frame, group_field="template_id", family=family, category_label="template", previous_records=previous_family_records),
            "strategy_type": _aggregate_group(frame, group_field="strategy_type", family=family, category_label="strategy_type", previous_records=previous_family_records),
            "idea_source": _aggregate_group(frame, group_field="idea_source", family=family, category_label="idea_source", previous_records=previous_family_records),
            "family_idea_kind": _aggregate_group(frame, group_field="idea_kind", family=family, category_label="family_idea_kind", previous_records=previous_family_records),
            "synthesized_template_family": _aggregate_group(frame, group_field="synthesized_template_family", family=family, category_label="synthesized_template_family", previous_records=previous_family_records),
            "idea_id": _aggregate_group(frame, group_field="idea_id", family=family, category_label="idea_id", previous_records=previous_family_records),
        }
        family_records = grouped_sources["idea_kind"]
        family_summary = _summarize_records(family_records)
        tg_groups = _aggregate_template_groups(frame, family=family)
        family_summary.update(build_family_idea_yield_guidance({
            **family_summary,
            "top_synthesized_template_families": [asdict(record) for record in sorted(grouped_sources["synthesized_template_family"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
        }))
        structural_frame = frame[frame["source_type"].astype(str) == "structural_synthesis"].copy()
        structural_llm_seeded_mask = structural_frame["source_idea_ids"].map(_has_non_empty_source_idea_ids) if "source_idea_ids" in structural_frame else pd.Series(dtype=bool)
        structural_execution_counts_by_family = {
            record.idea_key.split("::", 2)[-1]: int(record.idea_attempt_count)
            for record in grouped_sources["synthesized_template_family"]
        }
        family_summary.update(
            {
                "family": family,
                "recent_window": recent_window,
                "top_idea_kinds": [asdict(record) for record in sorted(grouped_sources["idea_kind"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_templates": [asdict(record) for record in sorted(grouped_sources["template_id"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_strategy_types": [asdict(record) for record in sorted(grouped_sources["strategy_type"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_idea_sources": [asdict(record) for record in sorted(grouped_sources["idea_source"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_family_idea_kinds": [asdict(record) for record in sorted(grouped_sources["family_idea_kind"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_synthesized_template_families": [asdict(record) for record in sorted(grouped_sources["synthesized_template_family"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "top_idea_ids": [asdict(record) for record in sorted(grouped_sources["idea_id"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]],
                "template_groups": tg_groups,
                "records": {
                    "idea_kind": {record.idea_key: asdict(record) for record in grouped_sources["idea_kind"]},
                    "template_id": {record.idea_key: asdict(record) for record in grouped_sources["template_id"]},
                    "strategy_type": {record.idea_key: asdict(record) for record in grouped_sources["strategy_type"]},
                    "idea_source": {record.idea_key: asdict(record) for record in grouped_sources["idea_source"]},
                    "family_idea_kind": {record.idea_key: asdict(record) for record in grouped_sources["family_idea_kind"]},
                    "synthesized_template_family": {record.idea_key: asdict(record) for record in grouped_sources["synthesized_template_family"]},
                    "idea_id": {record.idea_key: asdict(record) for record in grouped_sources["idea_id"]},
                },
                "structural_synthesis_proposal_count": int(len(structural_frame)),
                "structural_synthesis_execution_count": int(len(structural_frame)),
                "structural_synthesis_llm_seeded_count": int(structural_llm_seeded_mask.sum()) if not structural_llm_seeded_mask.empty else 0,
                "structural_synthesis_execution_counts_by_family": structural_execution_counts_by_family,
                "structural_synthesis_top_families": [
                    {
                        "synthesized_template_family": record.synthesized_template_family,
                        "idea_attempt_count": record.idea_attempt_count,
                        "idea_viable_count": record.idea_viable_count,
                        "idea_baseline_beat_count": record.idea_baseline_beat_count,
                        "idea_source": record.idea_source,
                        "idea_id": record.idea_id,
                    }
                    for record in sorted(grouped_sources["synthesized_template_family"], key=lambda record: (-record.idea_quality_score, -record.confidence, record.idea_key))[:5]
                ],
            }
        )
        family_summary["structural_family_feedback"] = {
            "old_state": family_summary.get("structural_family_state_previous"),
            "new_state": family_summary.get("structural_family_state"),
            "promotion_state": family_summary.get("structural_family_promotion_state"),
            "reason": family_summary.get("structural_family_transition_reason"),
            "graduated_template_family": family_summary.get("graduated_template_family"),
        }
        summary["families"][family] = family_summary
        summary["template_groups"].update(tg_groups)
        for group_name, records in grouped_sources.items():
            summary[f"by_{group_name}"] = {
                **summary.get(f"by_{group_name}", {}),
                **{record.idea_key: asdict(record) for record in records},
            }
            for record in records:
                idea_states.setdefault(family, {})[record.idea_key] = asdict(record)

    summary = _json_safe(summary)
    if persist_memory:
        save_research_memory(memory, base_dir)
    return summary


def save_idea_yield_summary(
    summary: dict[str, Any],
    *,
    base_dir: str = "experiments",
) -> Path:
    base_path = Path(base_dir)
    report_root = base_path.parent / "reports" if base_path.name == "experiments" else base_path / "reports"
    path = report_root / IDEA_YIELD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)
    return path


def load_latest_idea_yield_summary(base_dir: str = "experiments") -> dict[str, Any] | None:
    base_path = Path(base_dir)
    candidate_paths = []
    if base_path.name == "experiments":
        candidate_paths.append(base_path.parent / "reports" / IDEA_YIELD_FILENAME)
    else:
        candidate_paths.append(base_path / "reports" / IDEA_YIELD_FILENAME)
    candidate_paths.append(base_path / "scorecards" / IDEA_YIELD_FILENAME)
    candidate_paths.append(base_path.parent / "scorecards" / IDEA_YIELD_FILENAME)
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
