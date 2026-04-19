from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd


_EXPLICIT_REGIME_COLUMNS = (
    "regime",
    "regime_label",
    "market_regime",
    "vix_regime",
    "context_regime",
    "fear_greed_regime",
    "vix_state",
)


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


def _clean_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.map(_truthy).mean())


def _mean_float(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def _row_validation_strength(row: pd.Series) -> float:
    viable = 1.0 if _truthy(row.get("viable")) else 0.0
    baseline = 1.0 if _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails")) else 0.0
    sharpe = _clamp((_clean_float(row.get("sharpe"), 0.0) + 1.0) / 3.0)
    calmar = _clamp((_clean_float(row.get("calmar"), 0.0) + 1.0) / 4.0)
    total_return = _clamp((_clean_float(row.get("total_return"), 0.0) + 5.0) / 30.0)
    max_drawdown = _clean_float(row.get("max_drawdown"), None)
    drawdown = 0.5 if max_drawdown is None else _clamp(1.0 - abs(max_drawdown) / 40.0)
    trades = _clamp((_clean_float(row.get("trades_per_year"), 0.0)) / 40.0)
    exposure = _clean_float(row.get("exposure"), None)
    exposure_score = 0.5 if exposure is None else _clamp(exposure)
    negative_windows = _clean_float(row.get("negative_windows"), None)
    stability = 0.5 if negative_windows is None else _clamp(1.0 - abs(negative_windows) / 10.0)
    metric_profile = 0.35 * sharpe + 0.25 * calmar + 0.20 * total_return + 0.20 * drawdown
    activity_profile = 0.65 * trades + 0.35 * exposure_score
    return _clamp(
        0.25 * viable
        + 0.15 * baseline
        + 0.30 * metric_profile
        + 0.15 * activity_profile
        + 0.15 * stability
    )


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "timestamp_utc" in frame.columns:
        return frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    return frame.reset_index(drop=True)


def _slice_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = _ordered_frame(frame)
    total = len(ordered)
    if total == 0:
        return {"long": ordered, "medium": ordered, "short": ordered}
    if total < 3:
        return {"long": ordered.iloc[:1], "medium": ordered.iloc[:0], "short": ordered.iloc[1:]}
    long_end = max(1, total // 3)
    medium_end = max(long_end + 1, (2 * total) // 3)
    return {
        "long": ordered.iloc[:long_end],
        "medium": ordered.iloc[long_end:medium_end],
        "short": ordered.iloc[medium_end:],
    }


def _slice_strength(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    row_strengths = frame.apply(_row_validation_strength, axis=1)
    return _mean_float(row_strengths, default=0.0)


def _slice_tags(frame: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    slices = _slice_frame(frame)
    strengths = {name: _slice_strength(slice_frame) for name, slice_frame in slices.items()}
    counts = {name: len(slice_frame) for name, slice_frame in slices.items()}
    tags: list[str] = []
    if len(frame) < 3 or all(value is None for value in strengths.values()):
        return ["horizon_unknown"], {
            "slice_strengths": strengths,
            "slice_counts": counts,
            "coverage": 0.0,
            "confidence": 0.0,
            "scope": "unknown",
        }

    short_strength = strengths.get("short")
    medium_strength = strengths.get("medium")
    long_strength = strengths.get("long")

    strong_threshold = 0.65
    stable_threshold = 0.55
    weak_threshold = 0.45
    non_null_strengths = [value for value in strengths.values() if value is not None]
    if short_strength is not None and short_strength >= strong_threshold:
        tags.append("strong_short_horizon")
    if medium_strength is not None and medium_strength >= stable_threshold and (
        short_strength is None or abs(short_strength - medium_strength) <= 0.15
    ):
        tags.append("stable_medium_horizon")
    if long_strength is not None and long_strength >= stable_threshold and (
        short_strength is None or long_strength >= short_strength - 0.10
    ):
        tags.append("strong_long_horizon")
    if long_strength is not None and (long_strength <= weak_threshold or (short_strength is not None and short_strength - long_strength >= 0.20)):
        tags.append("weak_long_horizon")
    if non_null_strengths and (max(non_null_strengths) - min(non_null_strengths) >= 0.25):
        tags.append("horizon_mixed")
    if not tags:
        tags.append("horizon_unknown")

    coverage = sum(1 for value in strengths.values() if value is not None) / 3.0
    confidence = _clamp(min(1.0, len(frame) / 20.0) * coverage)
    strong_count = sum(1 for tag in tags if tag.startswith("strong_") or tag.startswith("stable_"))
    weak_count = sum(1 for tag in tags if tag.startswith("weak_"))
    if tags == ["horizon_unknown"]:
        scope = "unknown"
    elif strong_count >= 2 and weak_count == 0:
        scope = "broad"
    elif strong_count >= 1 and weak_count == 0 and "horizon_mixed" not in tags:
        scope = "partial"
    elif weak_count > 0 or "horizon_mixed" in tags:
        scope = "narrow"
    else:
        scope = "partial"

    return sorted(dict.fromkeys(tags)), {
        "slice_strengths": strengths,
        "slice_counts": counts,
        "coverage": round(coverage, 6),
        "confidence": round(confidence, 6),
        "scope": scope,
    }


def _normalize_regime_value(value: Any, row: pd.Series) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return None
    if any(token in text for token in ("bull", "recovery", "trend", "low_vol", "low volatility", "calm")):
        return "stable_in_trend"
    if any(token in text for token in ("bear", "drawdown", "crisis", "high_vol", "high volatility", "volatile")):
        if _truthy(row.get("viable")) and (_truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))):
            return "strong_in_bear"
        return "weak_in_high_vol"
    if any(token in text for token in ("mixed", "unknown")):
        return None
    return None


def _proxy_regime_tag(row: pd.Series) -> str | None:
    viable = _truthy(row.get("viable"))
    beats_baseline = _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))
    sharpe = _clean_float(row.get("sharpe"), 0.0) or 0.0
    calmar = _clean_float(row.get("calmar"), 0.0) or 0.0
    max_drawdown = abs(_clean_float(row.get("max_drawdown"), 0.0) or 0.0)
    trades = _clean_float(row.get("trades_per_year"), 0.0) or 0.0
    exposure = _clean_float(row.get("exposure"), None)
    negative_windows = _clean_float(row.get("negative_windows"), None)
    if viable and sharpe > 0.0 and calmar > 0.0 and max_drawdown <= 20.0 and trades >= 3.0 and (exposure is None or exposure >= 0.10):
        return "stable_in_trend"
    if viable and beats_baseline and (max_drawdown >= 15.0 or (negative_windows is not None and negative_windows >= 1.0)):
        return "strong_in_bear"
    if (not viable) or max_drawdown >= 25.0 or trades <= 3.0 or sharpe <= 0.0 or calmar <= 0.0:
        return "weak_in_high_vol"
    return None


def _regime_tags(frame: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    if frame.empty:
        return ["regime_unknown"], {"coverage": 0.0, "confidence": 0.0, "source_columns": [], "counts": {}}

    tags: list[str] = []
    counts: Counter[str] = Counter()
    source_columns = [column for column in _EXPLICIT_REGIME_COLUMNS if column in frame.columns]
    evidence_rows = 0
    for _, row in frame.iterrows():
        label: str | None = None
        for column in source_columns:
            label = _normalize_regime_value(row.get(column), row)
            if label:
                break
        if label is None:
            label = _proxy_regime_tag(row)
        if label is None:
            continue
        evidence_rows += 1
        counts[label] += 1
        tags.append(label)

    if not counts:
        return ["regime_unknown"], {
            "coverage": 0.0,
            "confidence": 0.0,
            "source_columns": source_columns,
            "counts": {},
        }

    unique_tags = set(counts)
    if len(unique_tags) > 1:
        tags.append("regime_mixed")
    coverage = evidence_rows / max(1, len(frame))
    confidence = _clamp(min(1.0, len(frame) / 20.0) * coverage)
    if "stable_in_trend" in unique_tags and "weak_in_high_vol" not in unique_tags and "strong_in_bear" not in unique_tags:
        scope = "broad"
    elif "weak_in_high_vol" in unique_tags and ("stable_in_trend" in unique_tags or "strong_in_bear" in unique_tags):
        scope = "partial"
    elif len(unique_tags) == 1:
        scope = "partial"
    else:
        scope = "narrow"

    return sorted(dict.fromkeys(tags)), {
        "coverage": round(coverage, 6),
        "confidence": round(confidence, 6),
        "source_columns": source_columns,
        "counts": dict(counts),
        "scope": scope,
    }


def summarize_validation_tags(frame: pd.DataFrame) -> dict[str, Any]:
    horizon_tags, horizon_details = _slice_tags(frame)
    regime_tags, regime_details = _regime_tags(frame)
    scope_priority = {"unknown": 0, "narrow": 1, "partial": 2, "broad": 3}
    horizon_scope = str(horizon_details.get("scope") or "unknown")
    regime_scope = str(regime_details.get("scope") or "unknown")
    if scope_priority.get(horizon_scope, 0) <= 0 and scope_priority.get(regime_scope, 0) <= 0:
        validation_scope = "unknown"
    elif scope_priority.get(horizon_scope, 0) >= 3 and scope_priority.get(regime_scope, 0) >= 2:
        validation_scope = "broad"
    elif "narrow" in {horizon_scope, regime_scope}:
        validation_scope = "narrow"
    elif "partial" in {horizon_scope, regime_scope}:
        validation_scope = "partial"
    else:
        validation_scope = "unknown"

    horizon_confidence = float(horizon_details.get("confidence") or 0.0)
    regime_confidence = float(regime_details.get("confidence") or 0.0)
    validation_confidence = _clamp((horizon_confidence + regime_confidence) / 2.0)
    validation_coverage = _clamp((float(horizon_details.get("coverage") or 0.0) + float(regime_details.get("coverage") or 0.0)) / 2.0)
    return {
        "validation_horizon_tags": horizon_tags,
        "validation_regime_tags": regime_tags,
        "validation_scope": validation_scope,
        "validation_confidence": round(validation_confidence, 6),
        "validation_coverage": round(validation_coverage, 6),
        "validation_horizon_details": horizon_details,
        "validation_regime_details": regime_details,
    }


def summarize_holdout_checks(
    *,
    validation_horizon_tags: list[str],
    validation_regime_tags: list[str],
    validation_scope: str,
    validation_confidence: float,
    validation_coverage: float,
    targeted_follow_up_type: str | None = None,
    holdout_check_type: str | None = None,
    holdout_check_status: str | None = None,
    holdout_check_outcome: str | None = None,
    holdout_check_scope: str | None = None,
    holdout_check_batch_id: str | None = None,
    holdout_horizon_tags: list[str] | None = None,
    holdout_regime_tags: list[str] | None = None,
    confirmation_outcome: str | None = None,
    promotion_state: str | None = None,
) -> dict[str, Any]:
    validation_horizon_tags = list(validation_horizon_tags or [])
    validation_regime_tags = list(validation_regime_tags or [])
    validation_scope = str(validation_scope or "unknown").strip().lower() or "unknown"
    confirmation_outcome = str(confirmation_outcome or "").strip().lower() or None
    promotion_state = str(promotion_state or "").strip().lower() or None
    validation_confidence = _clamp(float(validation_confidence or 0.0))
    validation_coverage = _clamp(float(validation_coverage or 0.0))
    holdout_check_type = str(holdout_check_type or "").strip() or None
    holdout_check_status = str(holdout_check_status or "").strip() or None
    holdout_check_outcome = str(holdout_check_outcome or "").strip() or None
    holdout_check_scope = str(holdout_check_scope or "").strip() or None
    holdout_check_batch_id = str(holdout_check_batch_id or "").strip() or None

    type_map = {
        "coverage_expansion_confirmation": ("coverage_expansion_holdout", "coverage_expansion"),
        "long_horizon_confirmation": ("long_horizon_holdout", "long_horizon"),
        "high_volatility_confirmation": ("high_volatility_holdout", "high_volatility"),
        "long_horizon_high_volatility_confirmation": ("long_horizon_high_volatility_holdout", "long_horizon_high_volatility"),
        "bear_regime_confirmation": ("bear_drawdown_holdout", "bear_drawdown"),
        "low_vol_trend_confirmation": ("trend_low_vol_holdout", "trend_low_vol"),
        "mixed_regime_clarification": ("mixed_regime_clarification_holdout", "mixed_regime"),
        "short_horizon_confirmation": ("short_horizon_holdout", "short_horizon"),
        "targeted_follow_up_confirmation": ("generic_holdout", "generic"),
    }

    inverse_type_map = {value[0]: value for value in type_map.values()}
    if holdout_check_type in inverse_type_map:
        holdout_check_type, holdout_check_scope = inverse_type_map[holdout_check_type]
    elif targeted_follow_up_type in type_map:
        holdout_check_type, holdout_check_scope = type_map[targeted_follow_up_type]
    else:
        if "weak_long_horizon" in validation_horizon_tags and "weak_in_high_vol" in validation_regime_tags:
            holdout_check_type, holdout_check_scope = "long_horizon_high_volatility_holdout", "long_horizon_high_volatility"
        elif "weak_long_horizon" in validation_horizon_tags:
            holdout_check_type, holdout_check_scope = "long_horizon_holdout", "long_horizon"
        elif "weak_in_high_vol" in validation_regime_tags:
            holdout_check_type, holdout_check_scope = "high_volatility_holdout", "high_volatility"
        elif "regime_mixed" in validation_regime_tags:
            holdout_check_type, holdout_check_scope = "mixed_regime_clarification_holdout", "mixed_regime"
        elif validation_scope in {"narrow", "partial", "unknown"}:
            holdout_check_type, holdout_check_scope = "coverage_expansion_holdout", "coverage_expansion"
        else:
            holdout_check_type, holdout_check_scope = "generic_holdout", "generic"

    derived_holdout_horizon_tags: list[str] = []
    if "horizon_unknown" in validation_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_horizon_unknown")
    if "strong_short_horizon" in validation_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_short_confirmed")
    if "stable_medium_horizon" in validation_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_medium_confirmed")
    if "strong_long_horizon" in validation_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_long_confirmed")
    if "weak_long_horizon" in validation_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_long_unproven")
    if not derived_holdout_horizon_tags:
        derived_holdout_horizon_tags.append("holdout_horizon_unproven" if validation_scope in {"narrow", "partial", "unknown"} else "holdout_horizon_confirmed")

    derived_holdout_regime_tags: list[str] = []
    if "regime_unknown" in validation_regime_tags:
        derived_holdout_regime_tags.append("holdout_regime_unknown")
    if "stable_in_trend" in validation_regime_tags:
        derived_holdout_regime_tags.append("holdout_trend_confirmed")
    if "strong_in_bear" in validation_regime_tags:
        derived_holdout_regime_tags.append("holdout_bear_confirmed")
    if "weak_in_high_vol" in validation_regime_tags:
        derived_holdout_regime_tags.append("holdout_high_vol_unproven")
    if "regime_mixed" in validation_regime_tags:
        derived_holdout_regime_tags.append("holdout_regime_mixed")
    if not derived_holdout_regime_tags:
        derived_holdout_regime_tags.append("holdout_regime_unproven" if validation_scope in {"narrow", "partial", "unknown"} else "holdout_regime_confirmed")

    if holdout_horizon_tags is None:
        holdout_horizon_tags = derived_holdout_horizon_tags
    if holdout_regime_tags is None:
        holdout_regime_tags = derived_holdout_regime_tags

    if holdout_check_status is None or holdout_check_outcome is None:
        if confirmation_outcome in {"confirmed", "passed", "success"}:
            holdout_check_status = "completed"
            holdout_check_outcome = "confirmed"
        elif confirmation_outcome in {"rejected", "failed"}:
            holdout_check_status = "completed"
            holdout_check_outcome = "rejected"
        elif validation_scope == "broad" and validation_confidence >= 0.60 and validation_coverage >= 0.60:
            holdout_check_status = "not_required"
            holdout_check_outcome = "broadly_confirmed"
        elif targeted_follow_up_type or validation_scope in {"narrow", "partial", "unknown"} or "weak_long_horizon" in validation_horizon_tags or "weak_in_high_vol" in validation_regime_tags or "regime_mixed" in validation_regime_tags:
            holdout_check_status = "required"
            holdout_check_outcome = "pending"
        else:
            holdout_check_status = "not_required"
            holdout_check_outcome = "unproven"

    if promotion_state in {"confirmed", "provisional", "rejected"} and holdout_check_status == "required":
        holdout_check_status = "partial" if promotion_state == "provisional" else holdout_check_status
    if holdout_check_type is None:
        holdout_check_type = "generic_holdout"
    if holdout_check_scope is None:
        holdout_check_scope = "generic"
    if holdout_check_batch_id is None:
        holdout_check_batch_id = None

    return {
        "holdout_check_type": holdout_check_type,
        "holdout_check_status": holdout_check_status,
        "holdout_check_outcome": holdout_check_outcome,
        "holdout_check_scope": holdout_check_scope,
        "holdout_check_batch_id": holdout_check_batch_id,
        "holdout_horizon_tags": sorted(dict.fromkeys(holdout_horizon_tags)),
        "holdout_regime_tags": sorted(dict.fromkeys(holdout_regime_tags)),
        "holdout_check_required": holdout_check_status in {"required", "partial"},
        "signals": {
            "validation_horizon_tags": validation_horizon_tags,
            "validation_regime_tags": validation_regime_tags,
            "validation_scope": validation_scope,
            "validation_confidence": validation_confidence,
            "validation_coverage": validation_coverage,
            "confirmation_outcome": confirmation_outcome,
            "promotion_state": promotion_state,
        },
    }


def build_validation_summaries(
    index: pd.DataFrame,
    families: list[str] | None = None,
    *,
    family_column: str = "strategy_family",
) -> dict[str, dict[str, Any]]:
    if index.empty or family_column not in index.columns:
        return {family: summarize_validation_tags(pd.DataFrame()) for family in (families or [])}

    available_families = families or sorted(
        str(family)
        for family in index[family_column].dropna().astype(str).tolist()
        if str(family).strip() and str(family).strip().lower() != "nan"
    )
    summaries: dict[str, dict[str, Any]] = {}
    for family in available_families:
        frame = index[index[family_column].astype(str) == family].copy()
        summaries[family] = summarize_validation_tags(frame)
    return summaries


def annotate_validation_tags(
    frame: pd.DataFrame,
    summaries: dict[str, dict[str, Any]],
    *,
    family_column: str = "strategy_family",
) -> pd.DataFrame:
    if frame.empty:
        return frame
    annotated = frame.copy()
    for column in (
        "validation_horizon_tags",
        "validation_regime_tags",
        "validation_scope",
        "validation_confidence",
        "validation_coverage",
    ):
        if column not in annotated.columns:
            annotated[column] = None
    if family_column not in annotated.columns:
        return annotated

    families = annotated[family_column].astype(str)
    for family, summary in summaries.items():
        mask = families == str(family)
        if not mask.any():
            continue
        for column in (
            "validation_horizon_tags",
            "validation_regime_tags",
            "validation_scope",
            "validation_confidence",
            "validation_coverage",
        ):
            if column in summary:
                value = summary.get(column)
                for idx in annotated.index[mask]:
                    annotated.at[idx, column] = value
    return annotated
