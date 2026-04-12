from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_dashboard import build_best_results_dashboard
from experiment_types import RuntimeDecision, RuntimeDecisionInput


DEFAULT_BASE_FAMILY_WEIGHTS = {
    "momentum": 0.65,
    "superstock": 0.20,
    "ml_ranker": 0.10,
    "rl_bandit": 0.05,
}


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return value
    return value


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_batch_overview(base_dir: str) -> dict[str, Any] | None:
    batch_root = Path(base_dir) / "batches"
    if not batch_root.exists():
        return None
    batch_dirs = sorted([path for path in batch_root.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)
    if not batch_dirs:
        return None
    batch_dir = batch_dirs[0]
    summary_path = batch_dir / "summary.json"
    summary = _load_json_file(summary_path) if summary_path.exists() else {}
    leaderboard_path = batch_dir / "leaderboard.csv"
    leaderboard_rows = 0
    if leaderboard_path.exists():
        try:
            leaderboard_rows = int(pd.read_csv(leaderboard_path).shape[0])
        except Exception:
            leaderboard_rows = 0
    return {
        "batch_id": batch_dir.name,
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "leaderboard_path": str(leaderboard_path) if leaderboard_path.exists() else None,
        "executed_count": summary.get("total_executed"),
        "sampled_count": summary.get("total_sampled"),
        "requested_count": summary.get("total_requested") or summary.get("requested"),
        "shortfall": summary.get("shortfall"),
        "status_counts": summary.get("status_counts") or {},
        "proposal_metadata": summary.get("proposal_metadata") or {},
        "leaderboard_rows": leaderboard_rows,
        "summary": summary,
    }


def _scorecard_dict(dashboard: dict[str, Any], family: str) -> dict[str, Any]:
    return dict((dashboard.get("family_scorecards") or {}).get(family) or {})


def _top_family_result(dashboard: dict[str, Any], family: str) -> dict[str, Any] | None:
    rows = (dashboard.get("top_per_family") or {}).get(family) or []
    if not rows:
        return None
    row = rows[0]
    return dict(row) if isinstance(row, dict) else dict(row or {})


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _overfitting_signals(
    *,
    family: str,
    scorecard: dict[str, Any],
    family_result: dict[str, Any] | None,
    best_overall: dict[str, Any] | None,
    best_viable: dict[str, Any] | None,
    best_baseline_beating: dict[str, Any] | None,
) -> dict[str, Any]:
    result = family_result or {}
    robustness_score = _safe_float(scorecard.get("robustness_score"), 0.0)
    scorecard_overfit_risk = _safe_float(scorecard.get("overfit_risk"), max(0.0, 1.0 - robustness_score))
    recent_robustness_trend = _safe_float(scorecard.get("recent_robustness_trend"), 0.0)
    risk = scorecard_overfit_risk
    flags: list[str] = []
    reasons: list[str] = []

    viable = _truthy(result.get("viable"))
    objective = _safe_float(result.get("objective_score"), 0.0)
    sharpe = _safe_float(result.get("sharpe"), 0.0)
    calmar = _safe_float(result.get("calmar"), 0.0)
    total_return = _safe_float(result.get("total_return"), 0.0)
    max_drawdown = abs(_safe_float(result.get("max_drawdown"), 0.0))
    trades_per_year = _safe_float(result.get("trades_per_year"), 0.0)
    exposure = result.get("exposure")
    exposure = None if exposure is None else _safe_float(exposure, 0.0)
    comparison_status = str(result.get("comparison_status") or "").strip()
    beats_objective = _truthy(result.get("beats_baseline_objective"))
    beats_guardrails = _truthy(result.get("beats_baseline_guardrails"))

    viable_rate = _safe_float(scorecard.get("viable_rate"), 0.0)
    recent_objective_trend = _safe_float(scorecard.get("recent_objective_trend"), 0.0)
    recent_viable_trend = _safe_float(scorecard.get("recent_viable_trend"), 0.0)
    dead_zone_density = _safe_float(scorecard.get("dead_zone_density"), 0.0)
    duplicate_saturation = _safe_float(scorecard.get("duplicate_saturation"), 0.0)
    confidence = _safe_float(scorecard.get("confidence"), 0.0)
    validation_horizon_tags = list(scorecard.get("validation_horizon_tags") or [])
    validation_regime_tags = list(scorecard.get("validation_regime_tags") or [])
    validation_scope = str(scorecard.get("validation_scope") or "unknown").strip().lower() or "unknown"
    validation_confidence = _safe_float(scorecard.get("validation_confidence"), 0.0)
    validation_coverage = _safe_float(scorecard.get("validation_coverage"), 0.0)
    stagnation = int(scorecard.get("stagnation_experiments") or 0)

    if not viable:
        risk += 0.25
        flags.append("not_viable")
        reasons.append("current winner is not viable")
    if objective >= 1.0 and not viable:
        risk += 0.15
        flags.append("high_objective_non_viable")
        reasons.append("high objective score without viability")
    if trades_per_year <= 3.0:
        risk += 0.15
        flags.append("sparse_trading")
        reasons.append("too few trades")
    if exposure is not None and exposure <= 0.10:
        risk += 0.10
        flags.append("sparse_exposure")
        reasons.append("too little exposure")
    if max_drawdown >= 25.0:
        risk += 0.10
        flags.append("large_drawdown")
        reasons.append("large drawdown")
    if sharpe <= 0.0:
        risk += 0.10
        flags.append("non_positive_sharpe")
        reasons.append("weak or negative Sharpe")
    if calmar <= 0.0:
        risk += 0.08
        flags.append("non_positive_calmar")
        reasons.append("weak or negative Calmar")
    if total_return <= 0.0:
        risk += 0.05
        flags.append("non_positive_return")
        reasons.append("weak or negative return")
    if not beats_objective and not beats_guardrails and comparison_status in {"exact_verified_current_engine", "partial_verified_current_engine"}:
        risk += 0.12
        flags.append("weak_baseline_comparison")
        reasons.append("baseline comparison is not strong")
    if viable_rate < 0.05:
        risk += 0.15
        flags.append("low_family_viable_rate")
        reasons.append("family has very low viable rate")
    if recent_objective_trend > 0.25 and viable_rate < 0.10:
        risk += 0.12
        flags.append("lucky_spike_pattern")
        reasons.append("objective trend looks like a lucky spike")
    if recent_viable_trend < -0.05:
        risk += 0.10
        flags.append("viable_trend_negative")
        reasons.append("recent viable trend is weakening")
    if recent_objective_trend < -0.10:
        risk += 0.08
        flags.append("objective_trend_negative")
        reasons.append("recent objective trend is weakening")
    if validation_scope == "broad":
        risk -= 0.08
        flags.append("broad_validation")
        reasons.append("broad validation tags support multiple horizons and regimes")
    elif validation_scope == "partial":
        risk -= 0.03
        flags.append("partial_validation")
        reasons.append("partial validation support is present")
    elif validation_scope == "narrow":
        risk += 0.08
        flags.append("narrow_validation")
        reasons.append("validation support is narrow")
    else:
        risk += 0.05
        flags.append("validation_unknown")
        reasons.append("validation coverage is unknown")
    if "strong_short_horizon" in validation_horizon_tags and "stable_medium_horizon" not in validation_horizon_tags and "strong_long_horizon" not in validation_horizon_tags:
        risk += 0.05
        flags.append("short_horizon_only")
        reasons.append("winner is only validated on the short horizon")
    if "weak_long_horizon" in validation_horizon_tags:
        risk += 0.08
        flags.append("weak_long_horizon")
        reasons.append("long-horizon validation is weak")
    if "stable_medium_horizon" in validation_horizon_tags:
        risk -= 0.03
        flags.append("stable_medium_horizon")
        reasons.append("medium-horizon validation is stable")
    if "strong_in_bear" in validation_regime_tags:
        risk -= 0.03
        flags.append("strong_in_bear")
        reasons.append("winner is robust in bear or drawdown-like regimes")
    if "stable_in_trend" in validation_regime_tags:
        risk -= 0.03
        flags.append("stable_in_trend")
        reasons.append("winner is stable in trend-like regimes")
    if "weak_in_high_vol" in validation_regime_tags:
        risk += 0.08
        flags.append("weak_in_high_vol")
        reasons.append("winner is weak in high-volatility or stress-like regimes")
    if "regime_mixed" in validation_regime_tags:
        risk += 0.04
        flags.append("regime_mixed")
        reasons.append("regime evidence is mixed")
    if dead_zone_density >= 0.65:
        risk += 0.12
        flags.append("dead_zone_density_high")
        reasons.append("dead-zone density is high")
    if duplicate_saturation >= 0.60:
        risk += 0.10
        flags.append("duplicate_saturation_high")
        reasons.append("duplicate saturation is high")
    if stagnation >= 50:
        risk += 0.08
        flags.append("stagnation_high")
        reasons.append("family has been stagnant for a long time")
    if family in {"rl_bandit", "ml_ranker"} and viable_rate <= 0.0 and dead_zone_density >= 0.50:
        risk += 0.10
        flags.append("weak_exploratory_family")
        reasons.append("exploratory family remains weak")
    if robustness_score >= 0.75:
        risk -= 0.08
        flags.append("robust_family")
        reasons.append("family-level robustness is strong")
    elif robustness_score >= 0.60:
        risk -= 0.04
        flags.append("moderately_robust_family")
        reasons.append("family-level robustness is acceptable")
    elif robustness_score <= 0.35:
        risk += 0.08
        flags.append("weak_family_robustness")
        reasons.append("family-level robustness is weak")
    if recent_robustness_trend > 0.05:
        risk -= 0.05
        flags.append("robustness_trend_positive")
        reasons.append("recent robustness trend is improving")
    elif recent_robustness_trend < -0.05:
        risk += 0.05
        flags.append("robustness_trend_negative")
        reasons.append("recent robustness trend is weakening")
    if family == (best_overall or {}).get("strategy_family") and risk < 0.35:
        risk -= 0.03
    if family == (best_viable or {}).get("strategy_family") and risk < 0.35:
        risk -= 0.03
    if family == (best_baseline_beating or {}).get("strategy_family") and risk < 0.35:
        risk -= 0.03

    risk = max(0.0, min(1.0, risk * max(0.25, 1.0 - min(confidence, 1.0) * 0.10)))
    risk = max(0.0, min(1.0, risk * max(0.35, 1.0 - min(validation_confidence, 1.0) * 0.12)))
    if validation_coverage < 0.35:
        flags.append("validation_coverage_low")
        reasons.append("validation coverage is sparse")
    do_not_over_exploit_yet = risk >= 0.45 or ("lucky_spike_pattern" in flags and risk >= 0.30)
    return {
        "family": family,
        "robustness_score": robustness_score,
        "scorecard_overfit_risk": scorecard_overfit_risk,
        "recent_robustness_trend": recent_robustness_trend,
        "validation_horizon_tags": validation_horizon_tags,
        "validation_regime_tags": validation_regime_tags,
        "validation_scope": validation_scope,
        "validation_confidence": validation_confidence,
        "validation_coverage": validation_coverage,
        "risk_score": round(risk, 6),
        "flags": flags,
        "reasons": reasons,
        "boosted_by_robustness": robustness_score >= 0.60,
        "downweighted_by_overfit": risk >= 0.45,
        "do_not_over_exploit_yet": do_not_over_exploit_yet,
        "signals": {
            "robustness_score": robustness_score,
            "overfit_risk": scorecard_overfit_risk,
            "recent_robustness_trend": recent_robustness_trend,
            "viable": viable,
            "objective_score": objective,
            "sharpe": sharpe,
            "calmar": calmar,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "trades_per_year": trades_per_year,
            "exposure": exposure,
            "comparison_status": comparison_status,
            "beats_baseline_objective": beats_objective,
            "beats_baseline_guardrails": beats_guardrails,
            "viable_rate": viable_rate,
            "recent_objective_trend": recent_objective_trend,
            "recent_viable_trend": recent_viable_trend,
            "dead_zone_density": dead_zone_density,
            "duplicate_saturation": duplicate_saturation,
            "stagnation_experiments": stagnation,
            "validation_horizon_tags": validation_horizon_tags,
            "validation_regime_tags": validation_regime_tags,
            "validation_scope": validation_scope,
            "validation_confidence": validation_confidence,
            "validation_coverage": validation_coverage,
        },
    }


def _winner_promotion_policy(
    *,
    family: str | None,
    scorecard: dict[str, Any],
    family_result: dict[str, Any] | None,
    overfit_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    if not family:
        return {
            "winner_family": None,
            "winner_promotion_status": "not_promoted",
            "winner_exploitation_cap": None,
            "confirmation_batch_requested": False,
            "reasons": ["no winner family available"],
            "signals": {},
        }

    result = family_result or {}
    signal = overfit_signal or {}
    robustness_score = _safe_float(scorecard.get("robustness_score"), 0.0)
    overfit_risk = _safe_float(scorecard.get("overfit_risk"), 1.0 - robustness_score)
    recent_robustness_trend = _safe_float(scorecard.get("recent_robustness_trend"), 0.0)
    viable_rate = _safe_float(scorecard.get("viable_rate"), 0.0)
    win_rate = _safe_float(scorecard.get("win_rate_vs_baseline"), 0.0)
    validation_horizon_tags = list(scorecard.get("validation_horizon_tags") or [])
    validation_regime_tags = list(scorecard.get("validation_regime_tags") or [])
    validation_scope = str(scorecard.get("validation_scope") or "unknown").strip().lower() or "unknown"
    validation_confidence = _safe_float(scorecard.get("validation_confidence"), 0.0)
    viable = _truthy(result.get("viable"))
    baseline_support = _truthy(result.get("beats_baseline_objective")) or _truthy(result.get("beats_baseline_guardrails")) or win_rate >= 0.50
    trades_per_year = _safe_float(result.get("trades_per_year"), 0.0)
    exposure = result.get("exposure")
    exposure = None if exposure is None else _safe_float(exposure, 0.0)
    sparse_activity = trades_per_year <= 3.0 or (exposure is not None and exposure <= 0.10)
    family_support_missing = viable_rate < 0.05 or (win_rate < 0.25 and not baseline_support)
    unstable_trend = recent_robustness_trend < -0.05 or _safe_float(scorecard.get("recent_viable_trend"), 0.0) < -0.05
    narrow_validation = validation_scope in {"narrow", "unknown"} or "weak_long_horizon" in validation_horizon_tags or "weak_in_high_vol" in validation_regime_tags
    broad_validation = validation_scope == "broad" and ("stable_in_trend" in validation_regime_tags or "strong_in_bear" in validation_regime_tags)
    suspicious = (
        overfit_risk >= 0.60
        or robustness_score < 0.50
        or not viable
        or sparse_activity
        or family_support_missing
        or unstable_trend
        or narrow_validation
        or bool(signal.get("do_not_over_exploit_yet"))
    )
    cautious = (
        not suspicious
        and (
            overfit_risk >= 0.35
            or robustness_score < 0.70
            or not baseline_support
            or viable_rate < 0.10
            or not broad_validation
            or validation_confidence < 0.50
        )
    )
    if suspicious:
        return {
            "winner_family": family,
            "winner_promotion_status": "hold_for_confirmation",
            "winner_exploitation_cap": 0.25,
            "confirmation_batch_requested": True,
            "reasons": [
                "winner is fragile or insufficiently supported",
                "confirmation run required before aggressive exploitation",
            ],
            "signals": {
                "robustness_score": robustness_score,
                "overfit_risk": overfit_risk,
                "viable_rate": viable_rate,
                "win_rate_vs_baseline": win_rate,
                "sparse_activity": sparse_activity,
                "family_support_missing": family_support_missing,
                "unstable_trend": unstable_trend,
                "baseline_support": baseline_support,
                "validation_horizon_tags": validation_horizon_tags,
                "validation_regime_tags": validation_regime_tags,
                "validation_scope": validation_scope,
                "validation_confidence": validation_confidence,
                "narrow_validation": narrow_validation,
            },
        }
    if cautious:
        return {
            "winner_family": family,
            "winner_promotion_status": "cautious_promotion",
            "winner_exploitation_cap": 0.40,
            "confirmation_batch_requested": False,
            "reasons": [
                "winner is promising but still needs restraint",
                "cap exploitation until repeatability improves",
            ],
            "signals": {
                "robustness_score": robustness_score,
                "overfit_risk": overfit_risk,
                "viable_rate": viable_rate,
                "win_rate_vs_baseline": win_rate,
                "sparse_activity": sparse_activity,
                "family_support_missing": family_support_missing,
                "unstable_trend": unstable_trend,
                "baseline_support": baseline_support,
                "validation_horizon_tags": validation_horizon_tags,
                "validation_regime_tags": validation_regime_tags,
                "validation_scope": validation_scope,
                "validation_confidence": validation_confidence,
                "narrow_validation": narrow_validation,
            },
        }
    return {
        "winner_family": family,
        "winner_promotion_status": "promoted",
        "winner_exploitation_cap": 0.60,
        "confirmation_batch_requested": False,
        "reasons": [
            "winner is robust, viable, and repeatable enough for normal promotion",
        ],
        "signals": {
            "robustness_score": robustness_score,
            "overfit_risk": overfit_risk,
            "viable_rate": viable_rate,
            "win_rate_vs_baseline": win_rate,
            "sparse_activity": sparse_activity,
            "family_support_missing": family_support_missing,
            "unstable_trend": unstable_trend,
            "baseline_support": baseline_support,
            "validation_horizon_tags": validation_horizon_tags,
            "validation_regime_tags": validation_regime_tags,
            "validation_scope": validation_scope,
            "validation_confidence": validation_confidence,
            "narrow_validation": narrow_validation,
        },
    }


def _confirmation_batch_plan(
    *,
    decision_id: str,
    request: RuntimeDecisionInput,
    winner_family: str | None,
    winner_scorecard: dict[str, Any],
    promotion_policy: dict[str, Any],
    latest_batch_overview: dict[str, Any] | None,
) -> dict[str, Any]:
    winner_promotion_status = str(promotion_policy.get("winner_promotion_status") or "not_promoted")
    confirmation_required = bool(promotion_policy.get("confirmation_batch_requested"))
    confirmation_reason_parts: list[str] = []
    robustness_score = _safe_float(winner_scorecard.get("robustness_score"), 0.0)
    overfit_risk = _safe_float(winner_scorecard.get("overfit_risk"), 1.0 - robustness_score)
    viable_rate = _safe_float(winner_scorecard.get("viable_rate"), 0.0)
    win_rate = _safe_float(winner_scorecard.get("win_rate_vs_baseline"), 0.0)
    confidence = _safe_float(winner_scorecard.get("confidence"), 0.0)
    recent_trend = _safe_float(winner_scorecard.get("recent_robustness_trend"), 0.0)
    validation_horizon_tags = list(winner_scorecard.get("validation_horizon_tags") or [])
    validation_regime_tags = list(winner_scorecard.get("validation_regime_tags") or [])
    validation_scope = str(winner_scorecard.get("validation_scope") or "unknown").strip().lower() or "unknown"
    validation_confidence = _safe_float(winner_scorecard.get("validation_confidence"), 0.0)

    if winner_promotion_status == "promoted" and confidence < 0.35:
        confirmation_required = True
        confirmation_reason_parts.append("promoted winner has limited evidence weight")
    if winner_promotion_status == "promoted" and (robustness_score < 0.70 or overfit_risk >= 0.25 or viable_rate < 0.20):
        confirmation_required = True
        confirmation_reason_parts.append("promoted winner still looks fragile enough to require proof")
    if winner_promotion_status == "cautious_promotion" and (robustness_score < 0.60 or overfit_risk >= 0.35):
        confirmation_required = True
        confirmation_reason_parts.append("cautious winner should be confirmed before over-exploitation")
    if winner_promotion_status == "hold_for_confirmation":
        confirmation_required = True
        confirmation_reason_parts.append("winner is already gated for confirmation")
    if win_rate < 0.50 and winner_family:
        confirmation_required = True
        confirmation_reason_parts.append("baseline-beating support is not yet strong enough")
    if validation_scope in {"narrow", "unknown"}:
        confirmation_required = True
        confirmation_reason_parts.append("winner needs targeted horizon/regime validation")
    if "weak_long_horizon" in validation_horizon_tags:
        confirmation_required = True
        confirmation_reason_parts.append("long-horizon validation is weak")
    if "weak_in_high_vol" in validation_regime_tags:
        confirmation_required = True
        confirmation_reason_parts.append("high-volatility validation is weak")
    if recent_trend < -0.05:
        confirmation_reason_parts.append("recent robustness trend is weakening")

    latest_meta = (latest_batch_overview or {}).get("proposal_metadata") or {}
    latest_confirmation_required = bool(latest_meta.get("confirmation_required"))
    latest_confirmation_state = str(latest_meta.get("confirmation_state") or "").strip().lower()
    latest_confirmation_outcome = str(latest_meta.get("confirmation_outcome") or "").strip().lower() or None
    if latest_confirmation_required and not latest_confirmation_outcome:
        executed_count = int((latest_batch_overview or {}).get("executed_count") or 0)
        requested_count = int((latest_batch_overview or {}).get("requested_count") or 0)
        total_failed = int((latest_batch_overview or {}).get("total_failed") or 0)
        if executed_count <= 0 or total_failed >= executed_count:
            latest_confirmation_outcome = "failed"
        elif requested_count and executed_count < requested_count:
            latest_confirmation_outcome = "provisional"
        else:
            latest_confirmation_outcome = "passed"
    if latest_confirmation_required:
        if latest_confirmation_outcome in {"confirmed", "passed", "success"}:
            promotion_state = "confirmed"
            confirmation_required = False
        elif latest_confirmation_outcome in {"rejected", "failed"}:
            promotion_state = "rejected"
            confirmation_required = False
        elif latest_confirmation_state in {"provisional", "confirmed", "rejected", "unconfirmed"}:
            promotion_state = latest_confirmation_state
        else:
            promotion_state = "provisional" if winner_promotion_status == "cautious_promotion" else "unconfirmed"
    elif winner_promotion_status == "promoted":
        promotion_state = "confirmed"
    elif winner_promotion_status == "cautious_promotion":
        promotion_state = "provisional"
    elif winner_promotion_status == "hold_for_confirmation":
        promotion_state = "unconfirmed"
    else:
        promotion_state = "unconfirmed"
    if confirmation_required and promotion_state == "confirmed":
        promotion_state = "unconfirmed" if winner_promotion_status == "promoted" else "provisional"

    if latest_confirmation_required and latest_confirmation_outcome in {"confirmed", "passed", "success"}:
        confirmation_required = False
        if promotion_state == "unconfirmed":
            promotion_state = "confirmed"
    confirmation_batch_id = f"{decision_id}_confirm_{winner_family}" if confirmation_required and winner_family else None
    planned_max_experiments = int(request.max_experiments)
    if confirmation_required:
        planned_max_experiments = max(1, min(int(request.max_experiments), max(4, int(round(int(request.max_experiments) * 0.25)))))
    confirmation_reason = "; ".join(dict.fromkeys([reason for reason in confirmation_reason_parts if reason]))
    if not confirmation_reason and confirmation_required:
        confirmation_reason = "winner needs a confirmation batch before it becomes a dominant refinement target"
    confirmation_family_budgets = None
    if confirmation_required and winner_family:
        if len(request.strategy_families or []) <= 1:
            confirmation_family_budgets = {winner_family: planned_max_experiments}
        else:
            winner_budget = max(1, int(round(planned_max_experiments * 0.70)))
            other_budget_total = max(0, planned_max_experiments - winner_budget)
            other_families = [family for family in request.strategy_families or [] if family != winner_family]
            confirmation_family_budgets = {winner_family: winner_budget}
            if other_families:
                per_other = max(0, other_budget_total // max(1, len(other_families)))
                for family in other_families:
                    confirmation_family_budgets[family] = max(0, per_other)
                assigned = sum(confirmation_family_budgets.values())
                if assigned < planned_max_experiments:
                    confirmation_family_budgets[winner_family] += planned_max_experiments - assigned

    return {
        "promotion_state": promotion_state,
        "confirmation_required": confirmation_required,
        "confirmation_reason": confirmation_reason,
        "confirmation_batch_id": confirmation_batch_id,
        "planned_max_experiments": planned_max_experiments,
        "confirmation_family_budgets": confirmation_family_budgets,
        "confirmation_outcome": latest_confirmation_outcome,
        "latest_confirmation_required": latest_confirmation_required,
        "validation_horizon_tags": validation_horizon_tags,
        "validation_regime_tags": validation_regime_tags,
        "validation_scope": validation_scope,
        "validation_confidence": validation_confidence,
    }


def _family_weight(
    family: str,
    *,
    scorecard: dict[str, Any],
    best_overall: dict[str, Any] | None,
    best_viable: dict[str, Any] | None,
    best_baseline_beating: dict[str, Any] | None,
    cycle_mode: str,
    overfit_signal: dict[str, Any] | None = None,
    promotion_policy: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    weight = DEFAULT_BASE_FAMILY_WEIGHTS.get(family, 0.05)
    reasons: list[str] = [f"base={weight:.2f}"]
    budget_stance = "active"
    robustness_score = float(scorecard.get("robustness_score") or 0.0)
    recent_robustness_trend = float(scorecard.get("recent_robustness_trend") or 0.0)
    overfit_risk = float(scorecard.get("overfit_risk") or max(0.0, 1.0 - robustness_score))
    viable_rate = float(scorecard.get("viable_rate") or 0.0)
    search_priority = float(scorecard.get("search_priority") or 0.0)
    confidence = float(scorecard.get("confidence") or 0.0)
    dead_zone_density = float(scorecard.get("dead_zone_density") or 0.0)
    duplicate_saturation = float(scorecard.get("duplicate_saturation") or 0.0)
    stagnation = int(scorecard.get("stagnation_experiments") or 0)
    validation_horizon_tags = list(scorecard.get("validation_horizon_tags") or [])
    validation_regime_tags = list(scorecard.get("validation_regime_tags") or [])
    validation_scope = str(scorecard.get("validation_scope") or "unknown").strip().lower() or "unknown"
    validation_confidence = _safe_float(scorecard.get("validation_confidence"), 0.0)

    if family == "momentum" and viable_rate > 0.05:
        weight += 0.10
        reasons.append("momentum_viable")
    if family == "momentum" and (best_baseline_beating or {}).get("strategy_family") == "momentum":
        weight += 0.22
        budget_stance = "primary"
        reasons.append("momentum_primary_baseline_beating")
    elif family == "momentum" and (best_viable or {}).get("strategy_family") == "momentum":
        weight += 0.12
        budget_stance = "primary"
        reasons.append("momentum_primary_viable")
    elif family == "momentum" and (best_overall or {}).get("strategy_family") == "momentum":
        weight += 0.08
        budget_stance = "primary"
        reasons.append("momentum_primary_overall")
    if robustness_score >= 0.80:
        weight += 0.12
        reasons.append("robustness_score_high")
    elif robustness_score >= 0.65:
        weight += 0.08
        reasons.append("robustness_score_good")
    elif robustness_score >= 0.50:
        weight += 0.04
        reasons.append("robustness_score_ok")
    if recent_robustness_trend > 0.05:
        weight += 0.05
        reasons.append("robustness_trend_positive")
    elif recent_robustness_trend < -0.05:
        weight *= 0.92
        reasons.append("robustness_trend_negative")
    if validation_scope == "broad":
        weight += 0.10
        reasons.append("validation_broad")
    elif validation_scope == "partial":
        weight += 0.04
        reasons.append("validation_partial")
    elif validation_scope == "narrow":
        weight *= 0.88
        reasons.append("validation_narrow")
    else:
        weight *= 0.85
        reasons.append("validation_unknown")
    if "strong_short_horizon" in validation_horizon_tags:
        weight += 0.03
        reasons.append("strong_short_horizon")
    if "stable_medium_horizon" in validation_horizon_tags:
        weight += 0.04
        reasons.append("stable_medium_horizon")
    if "strong_long_horizon" in validation_horizon_tags:
        weight += 0.05
        reasons.append("strong_long_horizon")
    if "weak_long_horizon" in validation_horizon_tags:
        weight *= 0.82
        reasons.append("weak_long_horizon")
    if "stable_in_trend" in validation_regime_tags:
        weight += 0.05
        reasons.append("stable_in_trend")
    if "strong_in_bear" in validation_regime_tags:
        weight += 0.04
        reasons.append("strong_in_bear")
    if "weak_in_high_vol" in validation_regime_tags:
        weight *= 0.80
        reasons.append("weak_in_high_vol")
    if "regime_mixed" in validation_regime_tags:
        weight *= 0.90
        reasons.append("regime_mixed")
    if validation_confidence < 0.35:
        weight *= 0.92
        reasons.append("validation_confidence_low")
    if family == (best_overall or {}).get("strategy_family"):
        weight += 0.15
        reasons.append("best_overall")
    if family == (best_viable or {}).get("strategy_family"):
        weight += 0.20
        reasons.append("best_viable")
    if family == (best_baseline_beating or {}).get("strategy_family"):
        weight += 0.25
        reasons.append("best_baseline_beating")
    if scorecard.get("recovery_signal"):
        weight += 0.10
        reasons.append("recovery_signal")
    if search_priority > 0:
        weight += search_priority * max(0.15, confidence) * 0.25
        reasons.append(f"search_priority={search_priority:.3f}")
    if viable_rate > 0.10:
        weight += 0.05
        reasons.append("viable_rate")
    if dead_zone_density >= 0.75:
        weight *= 0.60
        reasons.append("dead_zone_density_high")
    if duplicate_saturation >= 0.50:
        weight *= 0.75
        reasons.append("duplicate_saturation_high")
    if cycle_mode == "stagnation_escape" and stagnation >= 50:
        weight *= 1.10
        reasons.append("stagnation_escape")
    if cycle_mode == "confirmation":
        if family == (promotion_policy or {}).get("winner_family"):
            weight *= 1.55
            reasons.append("confirmation_focus")
        else:
            weight *= 0.55
            reasons.append("confirmation_restraint")
    if cycle_mode == "large-search":
        weight *= 1.05
        reasons.append("large_search_bias")
    if family == "superstock":
        if overfit_risk >= 0.80 or (robustness_score <= 0.55 and viable_rate < 0.20):
            weight *= 0.28
            budget_stance = "controlled"
            reasons.append("superstock_reduced_for_overfit_risk")
        elif overfit_risk >= 0.60 or viable_rate < 0.10:
            weight *= 0.55
            budget_stance = "controlled"
            reasons.append("superstock_reduced_for_stability")
    if family in {"rl_bandit", "ml_ranker"}:
        if viable_rate <= 0.0 and overfit_risk >= 0.65 and robustness_score <= 0.45 and not scorecard.get("recovery_signal"):
            weight *= 0.02
            budget_stance = "paused"
            reasons.append("family_paused_zero_viable_high_overfit")
        elif viable_rate <= 0.05 and overfit_risk >= 0.50:
            weight *= 0.10
            budget_stance = "paused"
            reasons.append("family_paused_low_viability_high_overfit")
        elif viable_rate <= 0.0 and dead_zone_density >= 0.50:
            weight *= 0.25
            budget_stance = "controlled"
            reasons.append("weak_exploratory_family")
    if promotion_policy and family == promotion_policy.get("winner_family"):
        status = str(promotion_policy.get("winner_promotion_status") or "not_promoted")
        if status == "promoted":
            weight *= 1.05
            reasons.append("winner_promoted")
        elif status == "cautious_promotion":
            weight *= 0.82
            reasons.append("winner_cautious_promotion")
        elif status == "hold_for_confirmation":
            weight *= 0.58
            reasons.append("winner_hold_for_confirmation")
        else:
            weight *= 0.75
            reasons.append("winner_not_promoted")
    overfit_risk = max(overfit_risk, float((overfit_signal or {}).get("risk_score") or 0.0))
    overfit_flags = list((overfit_signal or {}).get("flags") or [])
    if overfit_risk >= 0.70:
        weight *= 0.50
        reasons.append("overfit_risk_high")
    elif overfit_risk >= 0.45:
        weight *= 0.72
        reasons.append("overfit_risk_moderate")
    elif overfit_risk >= 0.25:
        weight *= 0.88
        reasons.append("overfit_risk_low")
    if overfit_flags and "lucky_spike_pattern" in overfit_flags:
        reasons.append("do_not_over_exploit_yet")
    return max(weight, 0.01), {
        "family": family,
        "budget_stance": budget_stance,
        "viable_rate": viable_rate,
        "search_priority": search_priority,
        "confidence": confidence,
        "dead_zone_density": dead_zone_density,
        "duplicate_saturation": duplicate_saturation,
        "stagnation_experiments": stagnation,
        "robustness_score": robustness_score,
        "overfit_risk": overfit_risk,
        "recent_robustness_trend": recent_robustness_trend,
        "validation_horizon_tags": validation_horizon_tags,
        "validation_regime_tags": validation_regime_tags,
        "validation_scope": validation_scope,
        "validation_confidence": validation_confidence,
        "validation_coverage": _safe_float(scorecard.get("validation_coverage"), 0.0),
        "overfit_flags": overfit_flags,
        "reasons": reasons,
    }


def _normalize_budgets(raw: dict[str, int], *, total: int, weights: dict[str, float]) -> dict[str, int]:
    budgets = {family: max(0, int(value)) for family, value in raw.items()}
    if not budgets:
        return {}
    while sum(budgets.values()) < total:
        family = max(budgets, key=lambda item: (weights.get(item, 0.0), -budgets[item], item))
        budgets[family] += 1
    while sum(budgets.values()) > total:
        removable = [family for family, value in budgets.items() if value > 0]
        if not removable:
            break
        family = min(removable, key=lambda item: (weights.get(item, 0.0), -budgets[item], item))
        budgets[family] -= 1
    return {family: value for family, value in budgets.items() if value > 0}


def _cycle_mode(
    *,
    selected_families: list[str],
    max_experiments: int,
    dashboard: dict[str, Any],
    latest_batch_overview: dict[str, Any] | None,
    stagnation_escape_batches: int,
    large_search_threshold: int,
    overfit_pressure: float = 0.0,
    overfit_family: str | None = None,
    promotion_policy: dict[str, Any] | None = None,
    confirmation_plan: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    best_overall = (dashboard.get("top_overall") or [{}])[0] if dashboard.get("top_overall") else None
    best_viable = (dashboard.get("top_viable") or [{}])[0] if dashboard.get("top_viable") else None
    best_baseline = (dashboard.get("top_baseline_beating") or [{}])[0] if dashboard.get("top_baseline_beating") else None
    scorecards = dashboard.get("family_scorecards") or {}
    best_overall_scorecard = scorecards.get((best_overall or {}).get("strategy_family")) or {}
    best_viable_scorecard = scorecards.get((best_viable or {}).get("strategy_family")) or {}
    best_baseline_scorecard = scorecards.get((best_baseline or {}).get("strategy_family")) or {}
    stagnation = max((int((scorecards.get(family) or {}).get("stagnation_experiments") or 0) for family in selected_families), default=0)
    executed = int((latest_batch_overview or {}).get("executed_count") or 0)
    requested = int((latest_batch_overview or {}).get("requested_count") or 0)
    shortfall = int((latest_batch_overview or {}).get("shortfall") or 0)
    underfilled = bool(requested and executed < requested)
    best_baseline_family = (best_baseline or {}).get("strategy_family")
    best_viable_family = (best_viable or {}).get("strategy_family")
    reasons = {
        "best_overall_family": (best_overall or {}).get("strategy_family"),
        "best_viable_family": best_viable_family,
        "best_baseline_beating_family": best_baseline_family,
        "stagnation_experiments": stagnation,
        "latest_batch_executed_count": executed,
        "latest_batch_requested_count": requested,
        "latest_batch_shortfall": shortfall,
        "latest_batch_underfilled": underfilled,
        "overfit_pressure": round(float(overfit_pressure), 6),
        "overfit_family": overfit_family,
        "best_overall_robustness_score": _safe_float(best_overall_scorecard.get("robustness_score"), 0.0),
        "best_viable_robustness_score": _safe_float(best_viable_scorecard.get("robustness_score"), 0.0),
        "best_baseline_robustness_score": _safe_float(best_baseline_scorecard.get("robustness_score"), 0.0),
        "best_overall_overfit_risk": _safe_float(best_overall_scorecard.get("overfit_risk"), 0.0),
        "best_viable_overfit_risk": _safe_float(best_viable_scorecard.get("overfit_risk"), 0.0),
        "best_baseline_overfit_risk": _safe_float(best_baseline_scorecard.get("overfit_risk"), 0.0),
        "winner_promotion_status": (promotion_policy or {}).get("winner_promotion_status"),
        "winner_exploitation_cap": (promotion_policy or {}).get("winner_exploitation_cap"),
        "confirmation_batch_requested": bool((promotion_policy or {}).get("confirmation_batch_requested")),
        "winner_family": (promotion_policy or {}).get("winner_family"),
        "promotion_state": (confirmation_plan or {}).get("promotion_state"),
        "confirmation_required": bool((confirmation_plan or {}).get("confirmation_required")),
        "confirmation_reason": (confirmation_plan or {}).get("confirmation_reason"),
    }
    if (confirmation_plan or {}).get("confirmation_required"):
        return "confirmation", reasons | {"confirmation_batch": True}
    if executed == 0 and requested > 0:
        return "diagnostics", reasons
    if stagnation >= stagnation_escape_batches * 20:
        return "stagnation_escape", reasons
    if shortfall > 0 or underfilled:
        return "large-search", reasons
    if max_experiments >= large_search_threshold:
        return "large-search", reasons
    if any((scorecards.get(family) or {}).get("recovery_signal") for family in selected_families):
        return "stagnation_escape", reasons
    if (promotion_policy or {}).get("winner_promotion_status") == "hold_for_confirmation":
        return "diagnostics", reasons | {"promotion_gate": "hold_for_confirmation"}
    if overfit_pressure >= 0.45:
        return "normal_exploration", reasons | {"anti_overfitting": True}
    if best_baseline_family in selected_families or best_viable_family in selected_families:
        if (promotion_policy or {}).get("winner_promotion_status") in {"promoted", "cautious_promotion"}:
            return "local_refinement", reasons
        return "normal_exploration", reasons | {"promotion_gate": (promotion_policy or {}).get("winner_promotion_status")}
    if (promotion_policy or {}).get("winner_promotion_status") in {"promoted", "cautious_promotion"}:
        return "local_refinement", reasons
    return "normal_exploration", reasons


def build_runtime_decision(request: RuntimeDecisionInput) -> RuntimeDecision:
    selected_families = [family.strip().lower() for family in (request.strategy_families or []) if family and family.strip()]
    max_experiments = int(request.max_experiments)
    timestamp = datetime.now(UTC).isoformat()
    decision_id = f"runtime_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    if not selected_families:
        return RuntimeDecision(
            decision_id=decision_id,
            timestamp_utc=timestamp,
            status="fallback",
            selected_families=[],
            cycle_mode="legacy_fallback",
            max_experiments=max_experiments,
            exploration_fraction=float(request.exploration_fraction),
            exploitation_fraction=float(request.exploitation_fraction),
            family_budgets=None,
            large_search_mode=False,
            min_large_search_candidates=int(request.min_large_search_candidates),
            dashboard_report_id=None,
            latest_batch_overview=None,
            latest_non_empty_batch=None,
            best_overall=None,
            best_viable=None,
            best_baseline_beating=None,
            family_scorecards={},
            used_signals={"fallback": True},
            rationale={"reason": "no strategy families selected"},
            fallback_used=True,
        )

    dashboard_obj = build_best_results_dashboard(base_dir=request.experiments_dir, families=selected_families)
    dashboard = _json_safe(asdict(dashboard_obj))
    if int((dashboard.get("counts") or {}).get("official_result_rows") or 0) <= 0:
        return RuntimeDecision(
            decision_id=decision_id,
            timestamp_utc=timestamp,
            status="fallback",
            selected_families=selected_families,
            cycle_mode="legacy_fallback",
            max_experiments=max_experiments,
            exploration_fraction=float(request.exploration_fraction),
            exploitation_fraction=float(request.exploitation_fraction),
            family_budgets=None,
            large_search_mode=False,
            min_large_search_candidates=int(request.min_large_search_candidates),
            dashboard_report_id=dashboard_obj.generated_at_utc,
            latest_batch_overview=None,
            latest_non_empty_batch=None,
            best_overall=None,
            best_viable=None,
            best_baseline_beating=None,
            family_scorecards={family: dict((dashboard.get("family_scorecards") or {}).get(family) or {}) for family in selected_families},
            used_signals={"dashboard_empty": True},
            rationale={
                "reason": "legacy fallback because the dashboard has no official results yet",
                "dashboard_report_id": dashboard_obj.generated_at_utc,
            },
            fallback_used=True,
        )
    latest_non_empty_batch = dashboard.get("latest_non_empty_batch")
    latest_batch_overview = _latest_batch_overview(request.experiments_dir)
    family_risk_reports: dict[str, Any] = {}
    for family in selected_families:
        family_risk_reports[family] = _overfitting_signals(
            family=family,
            scorecard=_scorecard_dict(dashboard, family),
            family_result=_top_family_result(dashboard, family),
            best_overall=(dashboard.get("top_overall") or [{}])[0] if dashboard.get("top_overall") else None,
            best_viable=(dashboard.get("top_viable") or [{}])[0] if dashboard.get("top_viable") else None,
            best_baseline_beating=(dashboard.get("top_baseline_beating") or [{}])[0] if dashboard.get("top_baseline_beating") else None,
        )
    focus_families = [
        family
        for family in [
            (dashboard.get("top_overall") or [{}])[0].get("strategy_family") if dashboard.get("top_overall") else None,
            (dashboard.get("top_viable") or [{}])[0].get("strategy_family") if dashboard.get("top_viable") else None,
            (dashboard.get("top_baseline_beating") or [{}])[0].get("strategy_family") if dashboard.get("top_baseline_beating") else None,
        ]
        if family in family_risk_reports
    ]
    if focus_families:
        overfit_pressure = max(float((family_risk_reports.get(family) or {}).get("risk_score") or 0.0) for family in focus_families)
        overfit_family = max(
            focus_families,
            key=lambda family: (
                float((family_risk_reports.get(family) or {}).get("risk_score") or 0.0),
                family,
            ),
        )
    else:
        overfit_pressure = 0.0
        overfit_family = None
    winner_family = None
    winner_source = None
    for source, rows in (
        ("baseline_beating", dashboard.get("top_baseline_beating") or []),
        ("best_viable", dashboard.get("top_viable") or []),
        ("best_overall", dashboard.get("top_overall") or []),
    ):
        if rows:
            winner_family = (rows[0] or {}).get("strategy_family")
            winner_source = source
            break
    winner_scorecard = _scorecard_dict(dashboard, winner_family) if winner_family else {}
    winner_result = _top_family_result(dashboard, winner_family) if winner_family else None
    winner_overfit_signal = family_risk_reports.get(winner_family) if winner_family else None
    promotion_policy = _winner_promotion_policy(
        family=winner_family,
        scorecard=winner_scorecard,
        family_result=winner_result,
        overfit_signal=winner_overfit_signal,
    )
    confirmation_plan = _confirmation_batch_plan(
        decision_id=decision_id,
        request=request,
        winner_family=winner_family,
        winner_scorecard=winner_scorecard,
        promotion_policy=promotion_policy,
        latest_batch_overview=latest_batch_overview,
    )
    cycle_mode, mode_signals = _cycle_mode(
        selected_families=selected_families,
        max_experiments=int(request.max_experiments),
        dashboard=dashboard,
        latest_batch_overview=latest_batch_overview,
        stagnation_escape_batches=int(request.stagnation_escape_batches),
        large_search_threshold=int(request.large_search_threshold),
        overfit_pressure=overfit_pressure,
        overfit_family=overfit_family,
        promotion_policy=promotion_policy,
        confirmation_plan=confirmation_plan,
    )

    exploration_fraction = float(request.exploration_fraction)
    if cycle_mode == "confirmation":
        exploration_fraction = max(exploration_fraction, 0.75)
    if cycle_mode == "stagnation_escape":
        exploration_fraction = max(exploration_fraction, 0.80)
    elif cycle_mode == "large-search":
        exploration_fraction = max(exploration_fraction, 0.70)
    elif cycle_mode == "diagnostics":
        exploration_fraction = max(exploration_fraction, 0.75)
    elif cycle_mode == "local_refinement":
        exploration_fraction = min(max(exploration_fraction, 0.45), 0.60)
    if overfit_pressure >= 0.45:
        exploration_fraction = max(exploration_fraction, 0.72)
    winner_exploitation_cap = _safe_float(promotion_policy.get("winner_exploitation_cap"), 0.35)
    exploitation_fraction = round(max(0.0, min(1.0, min(1.0 - exploration_fraction, winner_exploitation_cap))), 6)
    exploration_fraction = round(max(0.0, min(1.0, 1.0 - exploitation_fraction)), 6)

    family_scores: dict[str, float] = {}
    family_budgets: dict[str, int] = {}
    families_report: dict[str, Any] = {}
    best_overall = (dashboard.get("top_overall") or [{}])[0] if dashboard.get("top_overall") else None
    best_viable = (dashboard.get("top_viable") or [{}])[0] if dashboard.get("top_viable") else None
    best_baseline = (dashboard.get("top_baseline_beating") or [{}])[0] if dashboard.get("top_baseline_beating") else None
    scorecards = dashboard.get("family_scorecards") or {}
    for family in selected_families:
        scorecard = _scorecard_dict(dashboard, family)
        best_viable_family = (best_viable or {}).get("strategy_family")
        best_baseline_family = (best_baseline or {}).get("strategy_family")
        score, score_report = _family_weight(
            family,
            scorecard=scorecard,
            best_overall=best_overall,
            best_viable=best_viable,
            best_baseline_beating=best_baseline,
            cycle_mode=cycle_mode,
            overfit_signal=family_risk_reports.get(family),
            promotion_policy=promotion_policy,
        )
        family_scores[family] = score
        families_report[family] = score_report | {
            "scorecard": scorecard,
            "overfit_signal": family_risk_reports.get(family) or {},
            "selected": 0,
        }

    total_weight = sum(family_scores.values()) or float(len(selected_families))
    raw = {family: int(round(max_experiments * family_scores[family] / total_weight)) for family in selected_families}
    for family in selected_families:
        stance = str(families_report.get(family, {}).get("budget_stance") or "active")
        min_alive = 0 if stance == "paused" else 1
        if raw.get(family, 0) < min_alive:
            raw[family] = min_alive
    family_budgets = _normalize_budgets(raw, total=max_experiments, weights=family_scores)
    for family in selected_families:
        family_budgets.setdefault(family, 0)
    family_floors: dict[str, int] = {}
    for family in selected_families:
        stance = str(families_report.get(family, {}).get("budget_stance") or "active")
        family_floors[family] = 0 if stance == "paused" else 1
    for family, floor in family_floors.items():
        if family_budgets.get(family, 0) < floor:
            family_budgets[family] = floor
    total_assigned = sum(int(value) for value in family_budgets.values())
    while total_assigned > max_experiments:
        donor_candidates = [
            family
            for family in selected_families
            if family_budgets.get(family, 0) > family_floors.get(family, 0)
        ]
        if not donor_candidates:
            break
        donor = max(donor_candidates, key=lambda fam: (family_budgets.get(fam, 0), family_scores.get(fam, 0.0), fam))
        family_budgets[donor] -= 1
        total_assigned -= 1
    for family in selected_families:
        families_report.setdefault(family, {})["selected"] = family_budgets.get(family, 0)
        if family_budgets.get(family, 0) <= 0:
            if families_report[family].get("budget_stance") == "paused":
                families_report[family]["reason"] = (
                    "temporarily paused by dashboard evidence: zero viable rate, weak robustness, and high overfit risk"
                )
            else:
                families_report[family]["reason"] = "not retained after runtime budget normalization"
        elif family == "momentum" and family == best_baseline_family:
            families_report[family]["reason"] = "primary runtime refinement target because momentum remains the strongest baseline-aware family"
        elif family == best_viable_family:
            families_report[family]["reason"] = "runtime exploitation favored the current best viable family"
        else:
            families_report[family]["reason"] = "runtime budget allocated from dashboard scorecards, winners, losers, stagnation, and dead-zone signals"
        if families_report[family].get("budget_stance") == "primary":
            families_report[family]["reason"] = (
                families_report[family]["reason"]
                + "; promoted to primary runtime focus by baseline-beating and robustness evidence"
            )
        elif families_report[family].get("budget_stance") == "controlled":
            families_report[family]["reason"] = (
                families_report[family]["reason"]
                + "; heavily reduced because dashboard evidence shows high overfit risk or weak robustness"
            )
        elif families_report[family].get("budget_stance") == "paused":
            families_report[family]["reason"] = (
                families_report[family]["reason"]
                + "; temporarily paused because the family is not yet a reliable runtime target"
            )
        validation_scope = str(families_report[family].get("validation_scope") or "unknown").strip().lower() or "unknown"
        validation_horizon_tags = list(families_report[family].get("validation_horizon_tags") or [])
        validation_regime_tags = list(families_report[family].get("validation_regime_tags") or [])
        if validation_scope == "broad":
            families_report[family]["reason"] += "; validation is broad across horizons and regimes"
        elif validation_scope == "partial":
            families_report[family]["reason"] += "; validation is partial and should still be monitored"
        elif validation_scope == "narrow":
            families_report[family]["reason"] += "; validation is narrow, so exploitation remains cautious"
        else:
            families_report[family]["reason"] += "; validation coverage is unknown"
        if validation_horizon_tags:
            families_report[family]["reason"] += f"; horizon_tags={validation_horizon_tags}"
        if validation_regime_tags:
            families_report[family]["reason"] += f"; regime_tags={validation_regime_tags}"
        if families_report[family].get("robustness_score", 0.0) >= 0.70:
            families_report[family]["reason"] += "; boosted by robust family history"
        if families_report[family].get("overfit_risk", 0.0) >= 0.45:
            families_report[family]["reason"] += "; downweighted by overfit risk"
        if (family_risk_reports.get(family) or {}).get("do_not_over_exploit_yet"):
            families_report[family]["reason"] = (
                families_report[family]["reason"]
                + "; overfitting risk detected, so runtime exploitation was intentionally restrained"
            )
        if family == winner_family:
            families_report[family]["promotion_status"] = promotion_policy.get("winner_promotion_status")
            families_report[family]["exploitation_cap"] = promotion_policy.get("winner_exploitation_cap")
            families_report[family]["confirmation_batch_requested"] = confirmation_plan.get("confirmation_required")
            families_report[family]["winner_source"] = winner_source
        families_report[family]["confirmation_state"] = confirmation_plan.get("promotion_state")
        families_report[family]["confirmation_required"] = bool(confirmation_plan.get("confirmation_required"))
        if family == winner_family and confirmation_plan.get("confirmation_reason"):
            families_report[family]["confirmation_reason"] = confirmation_plan.get("confirmation_reason")
        if families_report[family].get("budget_stance") == "paused":
            families_report[family]["budget_cap"] = 0
        elif families_report[family].get("budget_stance") == "controlled":
            families_report[family]["budget_cap"] = max(1, family_budgets.get(family, 0))
        elif families_report[family].get("budget_stance") == "primary":
            families_report[family]["budget_cap"] = family_budgets.get(family, 0)

    planned_max_experiments = int(confirmation_plan.get("planned_max_experiments") or request.max_experiments)
    large_search_mode = (
        cycle_mode in {"large-search", "stagnation_escape", "diagnostics"}
        or (int(request.max_experiments) >= int(request.large_search_threshold) and not confirmation_plan.get("confirmation_required"))
    )
    adjusted_min_large = int(request.min_large_search_candidates)
    if large_search_mode and cycle_mode != "confirmation":
        adjusted_min_large = max(adjusted_min_large, min(int(request.max_experiments), max(48, int(round(int(request.max_experiments) * 0.50)))))
    elif cycle_mode == "confirmation":
        adjusted_min_large = max(4, min(adjusted_min_large, planned_max_experiments))
    winner_validation_horizon_tags = list(winner_scorecard.get("validation_horizon_tags") or [])
    winner_validation_regime_tags = list(winner_scorecard.get("validation_regime_tags") or [])
    winner_validation_scope = str(winner_scorecard.get("validation_scope") or "unknown").strip().lower() or "unknown"
    winner_validation_confidence = _safe_float(winner_scorecard.get("validation_confidence"), 0.0)
    winner_validation_coverage = _safe_float(winner_scorecard.get("validation_coverage"), 0.0)
    winner_validation_needs_follow_up = bool(
        winner_validation_scope in {"narrow", "unknown"}
        or "weak_long_horizon" in winner_validation_horizon_tags
        or "weak_in_high_vol" in winner_validation_regime_tags
        or (promotion_policy or {}).get("winner_promotion_status") == "hold_for_confirmation"
    )

    used_signals = {
        "best_overall": best_overall,
        "best_viable": best_viable,
        "best_baseline_beating": best_baseline,
        "latest_non_empty_batch": latest_non_empty_batch,
        "latest_batch_overview": latest_batch_overview,
        "family_scorecards": scorecards,
        "promotion_state": confirmation_plan.get("promotion_state"),
        "confirmation_required": bool(confirmation_plan.get("confirmation_required")),
        "confirmation_reason": confirmation_plan.get("confirmation_reason"),
        "confirmation_batch_id": confirmation_plan.get("confirmation_batch_id"),
        "planned_max_experiments": planned_max_experiments,
        "anti_overfitting": {
            "overfit_pressure": round(float(overfit_pressure), 6),
            "overfit_family": overfit_family,
            "family_risk_reports": family_risk_reports,
            "do_not_over_exploit_yet": bool(overfit_pressure >= 0.45),
            "robustness_signal_used": True,
            "winner_promotion_policy": promotion_policy,
        },
        "confirmation_plan": confirmation_plan,
        "dashboard_counts": dashboard.get("counts") or {},
        "mode_signals": mode_signals,
        "overfit_focus_families": focus_families,
        "winner_promotion_policy": promotion_policy,
        "winner_validation_horizon_tags": winner_validation_horizon_tags,
        "winner_validation_regime_tags": winner_validation_regime_tags,
        "winner_validation_scope": winner_validation_scope,
        "winner_validation_confidence": round(float(winner_validation_confidence), 6),
        "winner_validation_coverage": round(float(winner_validation_coverage), 6),
        "winner_validation_needs_follow_up": winner_validation_needs_follow_up,
    }
    rationale = {
        "reason": "dashboard and scorecard evidence drove the runtime decision, cycle mode, and family budget split",
        "cycle_mode": cycle_mode,
        "promotion_state": confirmation_plan.get("promotion_state"),
        "confirmation_required": bool(confirmation_plan.get("confirmation_required")),
        "confirmation_reason": confirmation_plan.get("confirmation_reason"),
        "confirmation_batch_id": confirmation_plan.get("confirmation_batch_id"),
        "planned_max_experiments": planned_max_experiments,
        "confirmation_family_budgets": confirmation_plan.get("confirmation_family_budgets"),
        "mode_signals": mode_signals,
        "family_budget_rationale": families_report,
        "anti_overfitting": {
            "overfit_pressure": round(float(overfit_pressure), 6),
            "overfit_family": overfit_family,
            "family_risk_reports": family_risk_reports,
            "do_not_over_exploit_yet": bool(overfit_pressure >= 0.45),
        },
        "exploration_fraction_reason": "broadened for large-search, diagnostics, or stagnation escape" if large_search_mode else "kept near default for targeted refinement",
        "used_dashboard_report_id": dashboard_obj.generated_at_utc,
        "latest_non_empty_batch": latest_non_empty_batch,
        "latest_batch_overview": latest_batch_overview,
        "overfit_focus_families": focus_families,
        "winner_promotion_policy": promotion_policy,
        "winner_family": winner_family,
        "winner_source": winner_source,
        "winner_validation_horizon_tags": winner_validation_horizon_tags,
        "winner_validation_regime_tags": winner_validation_regime_tags,
        "winner_validation_scope": winner_validation_scope,
        "winner_validation_confidence": round(float(winner_validation_confidence), 6),
        "winner_validation_coverage": round(float(winner_validation_coverage), 6),
        "winner_validation_needs_follow_up": winner_validation_needs_follow_up,
        "confirmation_plan": confirmation_plan,
    }

    return RuntimeDecision(
        decision_id=decision_id,
        timestamp_utc=timestamp,
        status="active",
        selected_families=selected_families,
        cycle_mode=cycle_mode,
        max_experiments=int(request.max_experiments),
        exploration_fraction=exploration_fraction,
        exploitation_fraction=exploitation_fraction,
        family_budgets=family_budgets,
        large_search_mode=large_search_mode,
        min_large_search_candidates=adjusted_min_large,
        dashboard_report_id=dashboard_obj.generated_at_utc,
        latest_batch_overview=latest_batch_overview,
        latest_non_empty_batch=latest_non_empty_batch,
        best_overall=best_overall,
        best_viable=best_viable,
        best_baseline_beating=best_baseline,
        family_scorecards={family: dict(scorecards.get(family) or {}) for family in selected_families},
        used_signals=used_signals,
        rationale=rationale,
        promotion_state=str(confirmation_plan.get("promotion_state") or "unconfirmed"),
        winner_family=winner_family,
        winner_promotion_status=str(promotion_policy.get("winner_promotion_status") or "not_promoted"),
        winner_exploitation_cap=_safe_float(promotion_policy.get("winner_exploitation_cap"), 0.35) if promotion_policy else None,
        winner_validation_horizon_tags=winner_validation_horizon_tags,
        winner_validation_regime_tags=winner_validation_regime_tags,
        winner_validation_scope=winner_validation_scope,
        winner_validation_confidence=winner_validation_confidence,
        winner_validation_coverage=winner_validation_coverage,
        winner_validation_needs_follow_up=winner_validation_needs_follow_up,
        confirmation_batch_requested=bool(confirmation_plan.get("confirmation_required")),
        confirmation_required=bool(confirmation_plan.get("confirmation_required")),
        confirmation_reason=confirmation_plan.get("confirmation_reason"),
        confirmation_batch_id=confirmation_plan.get("confirmation_batch_id"),
        confirmation_outcome=confirmation_plan.get("confirmation_outcome"),
        planned_max_experiments=planned_max_experiments,
        confirmation_family_budgets=confirmation_plan.get("confirmation_family_budgets"),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, prefix=path.name + ".", suffix=".tmp", encoding="utf-8") as handle:
        handle.write(content)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def save_runtime_decision(decision: RuntimeDecision, *, workspace_root: str = ".") -> Path:
    reports_dir = Path(workspace_root) / "reports" / "runtime_decisions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(decision))
    timestamped_path = reports_dir / f"{decision.decision_id}.json"
    latest_path = reports_dir / "latest.json"
    _atomic_write_text(timestamped_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    _atomic_write_text(latest_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return latest_path
