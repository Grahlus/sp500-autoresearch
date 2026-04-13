from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_best_results import rank_best_results
from experiment_lineage import build_lineage_summary
from experiment_memory import get_promotion_state_record, load_lineage_state_records, load_research_memory
from experiment_validation_tags import summarize_holdout_checks, summarize_validation_tags
from experiment_store import load_results_index


@dataclass(frozen=True)
class FamilyScorecard:
    family: str
    total_experiments: int
    viable_rate: float
    win_rate_vs_baseline: float | None
    robustness_score: float
    overfit_risk: float
    recent_robustness_trend: float
    best_objective_score: float | None
    median_objective_score: float | None
    recent_objective_trend: float
    recent_viable_trend: float
    dead_zone_density: float
    duplicate_saturation: float
    validation_horizon_tags: list[str]
    validation_regime_tags: list[str]
    validation_scope: str
    validation_confidence: float
    validation_coverage: float
    lineage_root_config_hash: str | None
    lineage_type: str
    lineage_depth: int
    descendant_count: int
    confirmation_descendant_count: int
    holdout_descendant_count: int
    rejected_descendant_count: int
    best_descendant_config_hash: str | None
    best_descendant_objective_score: float | None
    lineage_status_summary: str
    lineage_trust_score: float
    branch_balance: float
    holdout_check_required: bool
    holdout_check_type: str
    holdout_check_status: str
    holdout_check_outcome: str
    holdout_check_scope: str
    holdout_check_batch_id: str | None
    holdout_horizon_tags: list[str]
    holdout_regime_tags: list[str]
    promotion_state: str
    promotion_state_config_hash: str | None
    winner_promotion_status: str
    promotion_blocked_pending_new_evidence: bool
    promotion_state_reason: str | None
    confirmation_history_count: int
    holdout_history_count: int
    promotion_history_count: int
    last_confirmation_timestamp_utc: str | None
    last_confirmation_cycle_id: str | None
    last_failed_confirmation_timestamp_utc: str | None
    last_failed_confirmation_cycle_id: str | None
    last_improvement_timestamp: str | None
    exploration_budget_recommendation: float
    exploitation_budget_recommendation: float
    confidence: float
    evidence_weight: float
    stagnation_experiments: int
    recovery_signal: bool
    search_priority: float


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _clean_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.map(_truthy).mean())


def _median_float(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _trend(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0.0
    midpoint = len(numeric) // 2
    earlier = float(numeric.iloc[:midpoint].mean())
    later = float(numeric.iloc[midpoint:].mean())
    return later - earlier


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean_float(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def _latest_value(frame: pd.DataFrame, column: str) -> Any:
    if frame.empty or column not in frame.columns:
        return None
    values = frame[column].dropna()
    if values.empty:
        return None
    return values.iloc[-1]


def _listify(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item is not None and str(item).strip()]
            except Exception:
                pass
        return [text]
    return [str(value)]


def _bounded_metric_score(value: float | None, *, lower: float, upper: float) -> float:
    if value is None:
        return 0.5
    if upper <= lower:
        return 0.5
    return _clamp((float(value) - lower) / (upper - lower))


def _row_robustness_score(row: pd.Series) -> float:
    viable = 1.0 if _truthy(row.get("viable")) else 0.0
    baseline = 1.0 if _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails")) else 0.0
    sharpe = _bounded_metric_score(_clean_float(row.get("sharpe")), lower=-1.0, upper=2.0)
    calmar = _bounded_metric_score(_clean_float(row.get("calmar")), lower=-1.0, upper=3.0)
    total_return = _bounded_metric_score(_clean_float(row.get("total_return")), lower=-5.0, upper=25.0)
    max_drawdown = _clean_float(row.get("max_drawdown"))
    drawdown = 0.5 if max_drawdown is None else _clamp(1.0 - abs(max_drawdown) / 40.0)
    trades = _bounded_metric_score(_clean_float(row.get("trades_per_year")), lower=0.0, upper=40.0)
    exposure = _clean_float(row.get("exposure"))
    exposure_score = 0.5 if exposure is None else _clamp(exposure)
    negative_windows = _clean_float(row.get("negative_windows"))
    stability = 0.5 if negative_windows is None else _clamp(1.0 - abs(negative_windows) / 10.0)
    metric_profile = (
        0.35 * sharpe
        + 0.25 * calmar
        + 0.20 * total_return
        + 0.20 * drawdown
    )
    activity_profile = 0.65 * trades + 0.35 * exposure_score
    return _clamp(
        0.25 * viable
        + 0.15 * baseline
        + 0.30 * metric_profile
        + 0.15 * activity_profile
        + 0.15 * stability
    )


def _last_improvement_timestamp(frame: pd.DataFrame) -> tuple[str | None, int]:
    if frame.empty:
        return None, 0
    ordered = frame.sort_values("timestamp_utc")
    best = float("-inf")
    last_timestamp: str | None = None
    last_idx = -1
    for idx, (_, row) in enumerate(ordered.iterrows()):
        score = _clean_float(row.get("objective_score"))
        if score is None:
            continue
        if score > best:
            best = score
            last_timestamp = str(row.get("timestamp_utc")) if row.get("timestamp_utc") is not None else None
            last_idx = idx
    if last_idx < 0:
        return None, 0
    return last_timestamp, max(0, len(ordered) - last_idx - 1)


def _recommend_budget_mix(
    *,
    total_experiments: int,
    viable_rate: float,
    recent_objective_trend: float,
    recent_viable_trend: float,
    stagnation_experiments: int,
) -> tuple[float, float]:
    if total_experiments < 10:
        return 0.80, 0.20
    if recent_objective_trend > 0.10 and recent_viable_trend >= -0.05 and viable_rate >= 0.10:
        return 0.45, 0.55
    if stagnation_experiments >= 50 or viable_rate <= 0.02:
        return 0.75, 0.25
    return 0.60, 0.40


def _scorecard_from_frame(
    family: str,
    frame: pd.DataFrame,
    *,
    recent_window: int,
    promotion_state_record: dict[str, Any] | None = None,
    lineage_record: dict[str, Any] | None = None,
) -> FamilyScorecard:
    if frame.empty:
        validation = summarize_validation_tags(frame)
        exploration, exploitation = _recommend_budget_mix(
            total_experiments=0,
            viable_rate=0.0,
            recent_objective_trend=0.0,
            recent_viable_trend=0.0,
            stagnation_experiments=0,
        )
        return FamilyScorecard(
            family=family,
            total_experiments=0,
            viable_rate=0.0,
            win_rate_vs_baseline=None,
            robustness_score=0.0,
            overfit_risk=1.0,
            recent_robustness_trend=0.0,
            best_objective_score=None,
            median_objective_score=None,
            recent_objective_trend=0.0,
            recent_viable_trend=0.0,
            dead_zone_density=0.0,
            duplicate_saturation=0.0,
            validation_horizon_tags=list(validation["validation_horizon_tags"]),
            validation_regime_tags=list(validation["validation_regime_tags"]),
            validation_scope=str(validation["validation_scope"]),
            validation_confidence=_clamp(float(validation["validation_confidence"])),
            validation_coverage=_clamp(float(validation["validation_coverage"])),
            lineage_root_config_hash=(lineage_record or {}).get("lineage_root_config_hash"),
            lineage_type=str((lineage_record or {}).get("lineage_type") or "seed"),
            lineage_depth=int((lineage_record or {}).get("lineage_depth") or 0),
            descendant_count=int((lineage_record or {}).get("descendant_count") or 0),
            confirmation_descendant_count=int((lineage_record or {}).get("confirmation_descendant_count") or 0),
            holdout_descendant_count=int((lineage_record or {}).get("holdout_descendant_count") or 0),
            rejected_descendant_count=int((lineage_record or {}).get("rejected_descendant_count") or 0),
            best_descendant_config_hash=(lineage_record or {}).get("best_descendant_config_hash"),
            best_descendant_objective_score=(lineage_record or {}).get("best_descendant_objective_score"),
            lineage_status_summary=str((lineage_record or {}).get("lineage_status_summary") or "seed"),
            lineage_trust_score=_clamp(float((lineage_record or {}).get("lineage_trust_score") or 0.0)),
            branch_balance=_clamp(float((lineage_record or {}).get("branch_balance") or 0.0), lower=-1.0, upper=1.0),
            promotion_state=str((promotion_state_record or {}).get("promotion_state") or "unknown"),
            promotion_state_config_hash=(promotion_state_record or {}).get("config_hash"),
            winner_promotion_status=str((promotion_state_record or {}).get("winner_promotion_status") or "not_promoted"),
            promotion_blocked_pending_new_evidence=bool((promotion_state_record or {}).get("blocked_pending_new_evidence", False)),
            promotion_state_reason=(promotion_state_record or {}).get("block_reason"),
            confirmation_history_count=len((promotion_state_record or {}).get("confirmation_history") or []),
            holdout_history_count=len((promotion_state_record or {}).get("holdout_history") or []),
            promotion_history_count=len((promotion_state_record or {}).get("history") or []),
            last_confirmation_timestamp_utc=(promotion_state_record or {}).get("last_confirmation_timestamp_utc"),
            last_confirmation_cycle_id=(promotion_state_record or {}).get("last_confirmation_cycle_id"),
            last_failed_confirmation_timestamp_utc=(promotion_state_record or {}).get("last_failed_confirmation_timestamp_utc"),
            last_failed_confirmation_cycle_id=(promotion_state_record or {}).get("last_failed_confirmation_cycle_id"),
            holdout_check_required=False,
            holdout_check_type="generic_holdout",
            holdout_check_status="not_required",
            holdout_check_outcome="unproven",
            holdout_check_scope="generic",
            holdout_check_batch_id=None,
            holdout_horizon_tags=["holdout_horizon_unproven"],
            holdout_regime_tags=["holdout_regime_unproven"],
            last_improvement_timestamp=None,
            exploration_budget_recommendation=exploration,
            exploitation_budget_recommendation=exploitation,
            confidence=0.0,
            evidence_weight=0.0,
            stagnation_experiments=0,
            recovery_signal=False,
            search_priority=0.10,
        )

    total = int(len(frame))
    recent = frame.sort_values("timestamp_utc").tail(max(1, recent_window))
    viable_rate = _mean_bool(frame["viable"]) if "viable" in frame else 0.0
    recent_viable_trend = _trend(recent["viable"].map(lambda value: 1.0 if _truthy(value) else 0.0)) if "viable" in recent else 0.0
    recent_objective_trend = _trend(recent["objective_score"]) if "objective_score" in recent else 0.0
    ranked = rank_best_results(frame)
    best_objective = None
    if not ranked.empty:
        best_objective = float(ranked.iloc[0]["objective_score"])
    median_objective = _median_float(frame["objective_score"]) if "objective_score" in frame else None

    baseline_rows = frame[frame["beats_baseline_objective"].notna()] if "beats_baseline_objective" in frame else pd.DataFrame()
    win_rate = _mean_bool(baseline_rows["beats_baseline_objective"]) if not baseline_rows.empty else None

    row_robustness = frame.apply(_row_robustness_score, axis=1) if not frame.empty else pd.Series(dtype=float)
    recent_row_robustness = recent.apply(_row_robustness_score, axis=1) if not recent.empty else pd.Series(dtype=float)
    earlier = frame.sort_values("timestamp_utc").head(max(1, total // 2)) if total > 1 else pd.DataFrame()
    earlier_row_robustness = earlier.apply(_row_robustness_score, axis=1) if not earlier.empty else pd.Series(dtype=float)
    mean_row_robustness = _mean_float(row_robustness, default=0.0)
    recent_row_robustness_mean = _mean_float(recent_row_robustness, default=mean_row_robustness)
    earlier_row_robustness_mean = _mean_float(earlier_row_robustness, default=mean_row_robustness)
    recent_robustness_trend = recent_row_robustness_mean - earlier_row_robustness_mean
    validation = summarize_validation_tags(frame)
    validation_horizon_tags = list(validation["validation_horizon_tags"])
    validation_regime_tags = list(validation["validation_regime_tags"])
    validation_scope = str(validation["validation_scope"])
    validation_confidence = _clamp(float(validation["validation_confidence"]))
    validation_coverage = _clamp(float(validation["validation_coverage"]))
    latest_targeted_type = str(_latest_value(frame, "targeted_follow_up_type") or "").strip() or None
    latest_confirmation_outcome = str(_latest_value(frame, "confirmation_outcome") or "").strip() or None
    latest_confirmation_state = str(_latest_value(frame, "confirmation_state") or "").strip() or None
    latest_holdout_type = str(_latest_value(frame, "holdout_check_type") or "").strip() or None
    latest_holdout_status = str(_latest_value(frame, "holdout_check_status") or "").strip() or None
    latest_holdout_outcome = str(_latest_value(frame, "holdout_check_outcome") or "").strip() or None
    latest_holdout_scope = str(_latest_value(frame, "holdout_check_scope") or "").strip() or None
    latest_holdout_batch_id = str(_latest_value(frame, "holdout_check_batch_id") or "").strip() or None
    latest_holdout_horizon_tags = _listify(_latest_value(frame, "holdout_horizon_tags"))
    latest_holdout_regime_tags = _listify(_latest_value(frame, "holdout_regime_tags"))
    holdout = summarize_holdout_checks(
        validation_horizon_tags=validation_horizon_tags,
        validation_regime_tags=validation_regime_tags,
        validation_scope=validation_scope,
        validation_confidence=validation_confidence,
        validation_coverage=validation_coverage,
        targeted_follow_up_type=latest_targeted_type or latest_holdout_type,
        holdout_check_type=latest_holdout_type,
        holdout_check_status=latest_holdout_status,
        holdout_check_outcome=latest_holdout_outcome,
        holdout_check_scope=latest_holdout_scope,
        holdout_check_batch_id=latest_holdout_batch_id,
        holdout_horizon_tags=latest_holdout_horizon_tags or None,
        holdout_regime_tags=latest_holdout_regime_tags or None,
        confirmation_outcome=latest_confirmation_outcome or latest_holdout_outcome,
        promotion_state=latest_confirmation_state,
    )
    if latest_holdout_type:
        holdout["holdout_check_type"] = latest_holdout_type
    if latest_holdout_status:
        holdout["holdout_check_status"] = latest_holdout_status
    if latest_holdout_outcome:
        holdout["holdout_check_outcome"] = latest_holdout_outcome
    if latest_holdout_scope:
        holdout["holdout_check_scope"] = latest_holdout_scope
    if latest_holdout_batch_id:
        holdout["holdout_check_batch_id"] = latest_holdout_batch_id
    if latest_holdout_horizon_tags:
        holdout["holdout_horizon_tags"] = latest_holdout_horizon_tags
    if latest_holdout_regime_tags:
        holdout["holdout_regime_tags"] = latest_holdout_regime_tags
    lineage_root_config_hash = (lineage_record or {}).get("lineage_root_config_hash")
    lineage_type = str((lineage_record or {}).get("lineage_type") or "seed")
    lineage_depth = int((lineage_record or {}).get("lineage_depth") or 0)
    descendant_count = int((lineage_record or {}).get("descendant_count") or 0)
    confirmation_descendant_count = int((lineage_record or {}).get("confirmation_descendant_count") or 0)
    holdout_descendant_count = int((lineage_record or {}).get("holdout_descendant_count") or 0)
    rejected_descendant_count = int((lineage_record or {}).get("rejected_descendant_count") or 0)
    best_descendant_config_hash = (lineage_record or {}).get("best_descendant_config_hash")
    best_descendant_objective_score = (lineage_record or {}).get("best_descendant_objective_score")
    lineage_status_summary = str((lineage_record or {}).get("lineage_status_summary") or "seed")
    lineage_trust_score = _clamp(float((lineage_record or {}).get("lineage_trust_score") or 0.0))
    branch_balance = _clamp(float((lineage_record or {}).get("branch_balance") or 0.0), lower=-1.0, upper=1.0)
    promotion_state = str((promotion_state_record or {}).get("promotion_state") or latest_confirmation_state or "unconfirmed")
    if promotion_state == "not_promoted":
        promotion_state = "unconfirmed"
    winner_promotion_status = str((promotion_state_record or {}).get("winner_promotion_status") or "not_promoted")
    promotion_blocked_pending_new_evidence = bool((promotion_state_record or {}).get("blocked_pending_new_evidence", False))
    promotion_state_reason = (promotion_state_record or {}).get("block_reason")
    confirmation_history = list((promotion_state_record or {}).get("confirmation_history") or [])
    holdout_history = list((promotion_state_record or {}).get("holdout_history") or [])
    history = list((promotion_state_record or {}).get("history") or [])
    avg_trades = _mean_float(frame["trades_per_year"], default=0.0) if "trades_per_year" in frame else 0.0
    avg_exposure = _mean_float(frame["exposure"], default=0.5) if "exposure" in frame else 0.5
    activity_score = 0.65 * _clamp(avg_trades / 40.0) + 0.35 * _clamp(avg_exposure)
    baseline_score = 0.5 if win_rate is None else _clamp(win_rate)
    recent_score = _clamp((recent_row_robustness_mean + 1.0) / 2.0)
    historical_score = _clamp((mean_row_robustness + 1.0) / 2.0)
    confidence = min(1.0, total / 100.0)
    robustness_score = _clamp(
        0.38 * historical_score
        + 0.22 * recent_score
        + 0.18 * _clamp(viable_rate)
        + 0.12 * baseline_score
        + 0.10 * activity_score
    )
    robustness_score = _clamp(robustness_score * (0.85 + 0.15 * confidence))
    if validation_scope == "broad":
        robustness_score = _clamp(robustness_score + 0.05)
    elif validation_scope == "partial":
        robustness_score = _clamp(robustness_score + 0.02)
    elif validation_scope == "narrow":
        robustness_score = _clamp(robustness_score - 0.04)
    elif validation_scope == "unknown":
        robustness_score = _clamp(robustness_score - 0.05)
    if "stable_in_trend" in validation_regime_tags:
        robustness_score = _clamp(robustness_score + 0.03)
    if "strong_in_bear" in validation_regime_tags:
        robustness_score = _clamp(robustness_score + 0.03)
    if "weak_in_high_vol" in validation_regime_tags:
        robustness_score = _clamp(robustness_score - 0.05)

    dead_zone_values = pd.to_numeric(frame.get("dead_zone_risk", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    status_values = frame.get("status", pd.Series(dtype=str)).fillna("").astype(str)
    dead_zone_density = float(((dead_zone_values >= 0.50) | status_values.isin(["invalid", "no_trades"])).mean())

    duplicate_risk = frame.get("duplicate_risk", pd.Series(dtype=str)).notna()
    duplicate_status = status_values.eq("duplicate")
    unique_hashes = frame.get("config_hash", pd.Series(dtype=str)).dropna().astype(str).nunique()
    unique_saturation = 1.0 - (unique_hashes / max(1, total))
    duplicate_saturation = max(float((duplicate_risk | duplicate_status).mean()), float(unique_saturation))

    last_improvement, stagnation = _last_improvement_timestamp(frame)
    exploration, exploitation = _recommend_budget_mix(
        total_experiments=total,
        viable_rate=viable_rate,
        recent_objective_trend=recent_objective_trend,
        recent_viable_trend=recent_viable_trend,
        stagnation_experiments=stagnation,
    )
    evidence_weight = confidence * (0.50 + min(max(viable_rate, 0.0), 1.0))
    recovery_signal = bool(recent_objective_trend > 0.05 and recent_viable_trend >= 0.0)
    priority_base = 0.25
    if best_objective is not None:
        priority_base += max(min(best_objective, 3.0), -3.0) * 0.08
    priority = priority_base + viable_rate * 0.35 + recent_objective_trend * 0.10 - dead_zone_density * 0.15
    priority = max(0.01, min(1.0, priority))

    overfit_risk = _clamp(
        1.0 - robustness_score
        + dead_zone_density * 0.15
        + duplicate_saturation * 0.10
        + max(0.0, 0.20 - activity_score) * 0.40
        + (0.10 if viable_rate < 0.05 else 0.0)
        + (0.08 if baseline_score < 0.25 else 0.0)
        + (0.10 if recent_robustness_trend < -0.05 else 0.0)
        + (0.08 if recent_objective_trend < -0.10 else 0.0)
        + (0.08 if best_objective is not None and median_objective is not None and (best_objective - median_objective) > 0.75 and viable_rate < 0.35 else 0.0)
        + (0.06 if validation_scope == "narrow" else 0.0)
        + (0.08 if validation_scope == "unknown" else 0.0)
        + (0.06 if "weak_long_horizon" in validation_horizon_tags else 0.0)
        + (0.06 if "weak_in_high_vol" in validation_regime_tags else 0.0)
        + (0.04 if "regime_mixed" in validation_regime_tags else 0.0)
        - (0.05 if validation_scope == "broad" else 0.0)
        - (0.03 if "stable_in_trend" in validation_regime_tags else 0.0)
    )

    return FamilyScorecard(
        family=family,
        total_experiments=total,
        viable_rate=round(viable_rate, 6),
        win_rate_vs_baseline=None if win_rate is None else round(win_rate, 6),
        robustness_score=round(robustness_score, 6),
        overfit_risk=round(overfit_risk, 6),
        recent_robustness_trend=round(recent_robustness_trend, 6),
        best_objective_score=None if best_objective is None else round(best_objective, 6),
        median_objective_score=None if median_objective is None else round(median_objective, 6),
        recent_objective_trend=round(recent_objective_trend, 6),
        recent_viable_trend=round(recent_viable_trend, 6),
        dead_zone_density=round(dead_zone_density, 6),
        duplicate_saturation=round(duplicate_saturation, 6),
        validation_horizon_tags=validation_horizon_tags,
        validation_regime_tags=validation_regime_tags,
        validation_scope=validation_scope,
        validation_confidence=round(validation_confidence, 6),
        validation_coverage=round(validation_coverage, 6),
        lineage_root_config_hash=lineage_root_config_hash,
        lineage_type=lineage_type,
        lineage_depth=lineage_depth,
        descendant_count=descendant_count,
        confirmation_descendant_count=confirmation_descendant_count,
        holdout_descendant_count=holdout_descendant_count,
        rejected_descendant_count=rejected_descendant_count,
        best_descendant_config_hash=best_descendant_config_hash,
        best_descendant_objective_score=best_descendant_objective_score,
        lineage_status_summary=lineage_status_summary,
        lineage_trust_score=round(lineage_trust_score, 6),
        branch_balance=round(branch_balance, 6),
        holdout_check_required=bool(holdout["holdout_check_required"]),
        holdout_check_type=str(holdout["holdout_check_type"]),
        holdout_check_status=str(holdout["holdout_check_status"]),
        holdout_check_outcome=str(holdout["holdout_check_outcome"]),
        holdout_check_scope=str(holdout["holdout_check_scope"]),
        holdout_check_batch_id=holdout.get("holdout_check_batch_id"),
        holdout_horizon_tags=list(holdout["holdout_horizon_tags"]),
        holdout_regime_tags=list(holdout["holdout_regime_tags"]),
        promotion_state=promotion_state,
        promotion_state_config_hash=(promotion_state_record or {}).get("config_hash"),
        winner_promotion_status=winner_promotion_status,
        promotion_blocked_pending_new_evidence=promotion_blocked_pending_new_evidence,
        promotion_state_reason=promotion_state_reason,
        confirmation_history_count=len(confirmation_history),
        holdout_history_count=len(holdout_history),
        promotion_history_count=len(history),
        last_confirmation_timestamp_utc=(promotion_state_record or {}).get("last_confirmation_timestamp_utc"),
        last_confirmation_cycle_id=(promotion_state_record or {}).get("last_confirmation_cycle_id"),
        last_failed_confirmation_timestamp_utc=(promotion_state_record or {}).get("last_failed_confirmation_timestamp_utc"),
        last_failed_confirmation_cycle_id=(promotion_state_record or {}).get("last_failed_confirmation_cycle_id"),
        last_improvement_timestamp=last_improvement,
        exploration_budget_recommendation=round(exploration, 6),
        exploitation_budget_recommendation=round(exploitation, 6),
        confidence=round(confidence, 6),
        evidence_weight=round(evidence_weight, 6),
        stagnation_experiments=stagnation,
        recovery_signal=recovery_signal,
        search_priority=round(priority, 6),
    )


def build_family_scorecards(
    *,
    families: list[str],
    base_dir: str = "experiments",
    recent_window: int = 50,
) -> dict[str, FamilyScorecard]:
    index = load_results_index(base_dir)
    memory = load_research_memory(base_dir)
    lineage_summary = build_lineage_summary(index, persisted_records=load_lineage_state_records(base_dir), latest_batch=None)
    scorecards: dict[str, FamilyScorecard] = {}
    for family in families:
        if index.empty:
            frame = pd.DataFrame()
        else:
            frame = index[index["strategy_family"].astype(str) == family].copy()
        top_hash = None
        if not frame.empty and "config_hash" in frame.columns:
            ranked = rank_best_results(frame)
            if not ranked.empty:
                top_hash = str(ranked.iloc[0].get("config_hash") or "").strip() or None
        promotion_state_record = get_promotion_state_record(memory, family, top_hash) if top_hash else None
        lineage_record = None
        if top_hash:
            lineage_record = (lineage_summary.get("by_config_hash") or {}).get(top_hash)
        if lineage_record is None:
            lineage_record = (lineage_summary.get("family_summaries") or {}).get(family)
        scorecards[family] = _scorecard_from_frame(
            family,
            frame,
            recent_window=recent_window,
            promotion_state_record=promotion_state_record,
            lineage_record=lineage_record,
        )
    return scorecards


def scorecards_to_records(scorecards: dict[str, FamilyScorecard]) -> dict[str, dict[str, Any]]:
    return {family: asdict(scorecard) for family, scorecard in scorecards.items()}


def save_family_scorecards(
    scorecards: dict[str, FamilyScorecard],
    *,
    base_dir: str = "experiments",
    timestamp_utc: str | None = None,
) -> Path:
    path = Path(base_dir) / "scorecards" / "family_scorecards.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": timestamp_utc or datetime.now(UTC).isoformat(),
        "families": scorecards_to_records(scorecards),
    }
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)
    return path
