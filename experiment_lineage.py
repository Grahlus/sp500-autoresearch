from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from datetime import UTC, datetime
from typing import Any

import pandas as pd


_RESULT_STATUS_REJECTED = {"invalid", "no_trades", "error", "duplicate"}


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


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _safe_str(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _canonical_frame(index: pd.DataFrame) -> pd.DataFrame:
    if index.empty:
        return index.copy()
    frame = index.copy()
    for column in ("strategy_family", "config_hash"):
        if column not in frame.columns:
            frame[column] = None
    sort_columns = [column for column in ("timestamp_utc", "experiment_id") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort")
    frame["strategy_family"] = frame["strategy_family"].astype(str)
    frame["config_hash"] = frame["config_hash"].astype(str)
    family_mask = ~frame["strategy_family"].str.strip().str.lower().isin({"", "nan", "none"})
    hash_mask = ~frame["config_hash"].str.strip().str.lower().isin({"", "nan", "none"})
    frame = frame[family_mask & hash_mask]
    frame = frame.drop_duplicates(subset=["strategy_family", "config_hash"], keep="last").reset_index(drop=True)
    return frame


def infer_lineage_type(row: pd.Series) -> str:
    source_type = _safe_str(row.get("source_type"), "").lower() or ""
    proposal_role = _safe_str(row.get("proposal_role"), "").lower() or ""
    exploration_mode = _safe_str(row.get("exploration_mode"), "").lower() or ""
    strategy_type = _safe_str(row.get("strategy_type"), "").lower() or ""
    confirmation_required = _truthy(row.get("confirmation_required")) or _safe_str(row.get("confirmation_state"), "").lower() in {
        "unconfirmed",
        "provisional",
        "confirmed",
        "rejected",
    }
    holdout_required = _truthy(row.get("holdout_check_required")) or bool(_safe_str(row.get("holdout_check_type"), None))
    if confirmation_required or _safe_str(row.get("confirmation_batch_id"), None) or _safe_str(row.get("confirmation_trial_kind"), None):
        return "confirmation"
    if holdout_required or _safe_str(row.get("holdout_check_batch_id"), None):
        return "holdout"
    if source_type == "cross_family_hybrid" or proposal_role == "cross_family_hybrid":
        return "cross_family_hybrid"
    if source_type in {"broad_exploration", "fallback_random", "idea_seed", "analysis_seed"} or exploration_mode in {
        "broader_exploration",
        "large_search",
        "fallback_random",
    }:
        return "exploration_branch"
    if proposal_role in {"explore", "exploration", "broader_exploration"}:
        return "exploration_branch"
    if source_type in {"classical", "ml", "rl"} or strategy_type in {"classical", "ml", "rl"}:
        return "local_refinement" if _safe_str(row.get("parent_config_hash"), None) else "seed"
    if _safe_str(row.get("parent_config_hash"), None):
        return "local_refinement"
    return "seed"


def _is_holdout_descendant(row: pd.Series) -> bool:
    holdout_required = _truthy(row.get("holdout_check_required")) or bool(_safe_str(row.get("holdout_check_type"), None))
    holdout_outcome = str(row.get("holdout_check_outcome") or "").strip().lower()
    holdout_status = str(row.get("holdout_check_status") or "").strip().lower()
    return bool(
        holdout_required
        or _safe_str(row.get("holdout_check_batch_id"), None)
        or holdout_outcome in {"confirmed", "broadly_confirmed", "passed", "success", "rejected", "failed", "unproven", "pending", "provisional"}
        or holdout_status in {"required", "partial", "confirmed", "rejected"}
    )


def _lineage_status_summary(
    *,
    lineage_type: str,
    descendant_count: int,
    confirmation_descendant_count: int,
    holdout_descendant_count: int,
    rejected_descendant_count: int,
    parent_objective_score: float | None,
    best_descendant_objective_score: float | None,
) -> str:
    support_count = confirmation_descendant_count + holdout_descendant_count
    if descendant_count <= 0:
        return "seed" if lineage_type == "seed" else "leaf"
    if rejected_descendant_count >= max(2, support_count) and rejected_descendant_count >= max(1, descendant_count // 2):
        return "failing"
    improving = (
        parent_objective_score is not None
        and best_descendant_objective_score is not None
        and best_descendant_objective_score > parent_objective_score
    )
    if support_count > rejected_descendant_count or improving:
        return "strengthening"
    if rejected_descendant_count > 0 and support_count > 0:
        return "mixed"
    return "leaf"


def _lineage_trust_score(
    *,
    descendant_count: int,
    confirmation_descendant_count: int,
    holdout_descendant_count: int,
    rejected_descendant_count: int,
    parent_objective_score: float | None,
    best_descendant_objective_score: float | None,
    lineage_type: str,
) -> float:
    if descendant_count <= 0:
        return 0.10 if lineage_type == "seed" else 0.20
    support_ratio = (confirmation_descendant_count + holdout_descendant_count) / max(1, descendant_count)
    rejection_ratio = rejected_descendant_count / max(1, descendant_count)
    improvement_bonus = 0.0
    if parent_objective_score is not None and best_descendant_objective_score is not None:
        delta = best_descendant_objective_score - parent_objective_score
        if delta > 0:
            improvement_bonus = min(0.20, delta * 0.08)
        elif delta < 0:
            improvement_bonus = max(-0.15, delta * 0.06)
    lineage_type_bonus = {
        "confirmation": 0.10,
        "holdout": 0.08,
        "local_refinement": 0.06,
        "cross_family_hybrid": 0.04,
        "exploration_branch": 0.02,
        "seed": 0.00,
    }.get(lineage_type, 0.0)
    score = 0.18 + 0.42 * support_ratio - 0.30 * rejection_ratio + improvement_bonus + lineage_type_bonus
    return max(0.0, min(1.0, score))


def _branch_balance(
    *,
    descendant_count: int,
    confirmation_descendant_count: int,
    holdout_descendant_count: int,
    rejected_descendant_count: int,
) -> float:
    if descendant_count <= 0:
        return 0.0
    support = confirmation_descendant_count + holdout_descendant_count
    return max(-1.0, min(1.0, (support - rejected_descendant_count) / max(1, descendant_count)))


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    if isinstance(parsed, datetime):
        return parsed
    return None


def _days_between(start: Any, end: Any) -> int:
    start_dt = _parse_timestamp(start)
    end_dt = _parse_timestamp(end)
    if start_dt is None or end_dt is None:
        return 0
    delta = end_dt - start_dt
    return max(0, int(delta.total_seconds() // 86400))


def _branch_decay_state(
    *,
    lineage_type: str,
    lineage_status_summary: str,
    lineage_trust_score: float,
    branch_balance: float,
    descendant_count: int,
    confirmation_descendant_count: int,
    holdout_descendant_count: int,
    rejected_descendant_count: int,
    invalid_descendant_count: int,
    no_trades_descendant_count: int,
    failed_confirmation_descendant_count: int,
    failed_holdout_descendant_count: int,
    targeted_follow_up_failure_count: int,
    branch_age_days: int,
    best_descendant_objective_score: float | None,
    parent_objective_score: float | None,
    previous_branch_state: str | None = None,
    previous_branch_decay_score: float | None = None,
    previous_best_descendant_objective_score: float | None = None,
) -> tuple[float, str, str, str | None, bool, str | None]:
    support_count = confirmation_descendant_count + holdout_descendant_count
    rejection_count = rejected_descendant_count + failed_confirmation_descendant_count + failed_holdout_descendant_count
    failure_density = (
        invalid_descendant_count + no_trades_descendant_count + rejection_count + targeted_follow_up_failure_count
    ) / max(1, descendant_count)
    staleness_penalty = _clamp(branch_age_days / 30.0)
    improvement_delta = 0.0
    if parent_objective_score is not None and best_descendant_objective_score is not None:
        improvement_delta = best_descendant_objective_score - parent_objective_score
    prior_improvement_delta = 0.0
    if previous_best_descendant_objective_score is not None and best_descendant_objective_score is not None:
        prior_improvement_delta = best_descendant_objective_score - previous_best_descendant_objective_score
    decay_score = _clamp(
        0.50 * (1.0 - lineage_trust_score)
        + 0.16 * failure_density
        + 0.10 * _clamp(invalid_descendant_count / max(1, descendant_count))
        + 0.08 * _clamp(no_trades_descendant_count / max(1, descendant_count))
        + 0.08 * staleness_penalty
        + 0.05 * max(0.0, -branch_balance)
        + (0.05 if lineage_status_summary == "failing" else 0.0)
        + (0.03 if lineage_status_summary == "mixed" else 0.0)
        - (0.05 if lineage_status_summary == "strengthening" else 0.0)
        - (0.04 if improvement_delta > 0.0 else 0.0)
        - (0.03 if support_count > rejection_count else 0.0)
    )
    recovery_signal = bool(
        previous_branch_state in {"paused", "retired"}
        and (
            prior_improvement_delta > 0.05
            or improvement_delta > 0.05
            or support_count > max(0, rejection_count)
            or (
                lineage_status_summary == "strengthening"
                and previous_branch_decay_score is not None
                and decay_score + 0.10 < previous_branch_decay_score
            )
        )
    )
    if recovery_signal:
        branch_state = "revivable"
    elif decay_score >= 0.82 or (failure_density >= 0.70 and branch_age_days >= 21 and support_count == 0):
        branch_state = "retired"
    elif decay_score >= 0.62 or failure_density >= 0.50 or (lineage_status_summary == "failing" and branch_age_days >= 14):
        branch_state = "paused"
    elif decay_score >= 0.40:
        branch_state = "decaying"
    else:
        branch_state = "active"
    if branch_state == "active" and previous_branch_state in {"paused", "retired"} and recovery_signal:
        branch_state = "revivable"

    if branch_state == "retired":
        retirement_reason = "repeated failures, sparse activity, or staleness made the branch non-competitive"
    elif branch_state == "paused":
        retirement_reason = "branch is decaying from repeated failures or stale evidence"
    elif branch_state == "decaying":
        retirement_reason = "branch is losing support and should be budget-reduced"
    else:
        retirement_reason = None

    if branch_state == "revivable":
        revival_condition = "new evidence improved this branch; keep a small controlled budget and require repeat success"
    elif branch_state in {"paused", "retired"}:
        revival_condition = "requires meaningful new improvement, fewer failures, and a fresh confirmation or holdout pass"
    else:
        revival_condition = "no revival needed"

    return decay_score, branch_state, lineage_status_summary, retirement_reason, recovery_signal, revival_condition


def _branch_state_multiplier(branch_state: str) -> float:
    return {
        "active": 1.0,
        "revivable": 0.65,
        "decaying": 0.45,
        "paused": 0.08,
        "retired": 0.0,
    }.get(str(branch_state or "").strip().lower(), 0.3)


def _sort_numeric(value: Any) -> float:
    if value is None:
        return float("-inf")
    try:
        if pd.isna(value):
            return float("-inf")
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _sort_descendants(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    sortable = frame.copy()
    for column in ("objective_score", "sharpe", "calmar", "total_return"):
        if column in sortable.columns:
            sortable[column] = pd.to_numeric(sortable[column], errors="coerce")
    if "viable" in sortable.columns:
        sortable["viable"] = sortable["viable"].map(_truthy)
    sort_columns = [column for column in ("viable", "objective_score", "sharpe", "calmar", "total_return", "timestamp_utc", "experiment_id") if column in sortable.columns]
    ascending = [False, False, False, False, False, True, True][: len(sort_columns)]
    return sortable.sort_values(by=sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)


def build_lineage_summary(
    index: pd.DataFrame,
    *,
    persisted_records: dict[str, dict[str, Any]] | None = None,
    latest_batch: dict[str, Any] | None = None,
    include_histories: bool = True,
    include_records: bool = True,
) -> dict[str, Any]:
    canonical = _canonical_frame(index)
    if canonical.empty:
        return {
            "result_count": 0,
            "latest_batch_id": latest_batch.get("batch_id") if latest_batch else None,
            "records": {} if include_records else {},
            "by_config_hash": {},
            "family_summaries": {},
            "branch_summaries": {},
            "lineage_status_counts": {},
        }

    row_map: dict[str, dict[str, Any]] = {}
    family_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for _, row in canonical.iterrows():
        record = _json_safe(row.to_dict())
        family = str(record.get("strategy_family") or "").strip().lower()
        config_hash = str(record.get("config_hash") or "").strip()
        if not family or not config_hash:
            continue
        row_map[config_hash] = record
        family_map[family][config_hash] = record

    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for config_hash, record in row_map.items():
        parent_hash = _safe_str(record.get("parent_config_hash"), None)
        if parent_hash and parent_hash in row_map and parent_hash != config_hash:
            children_by_parent[parent_hash].append(config_hash)

    @lru_cache(maxsize=None)
    def _root_for(config_hash: str) -> str:
        record = row_map.get(config_hash) or {}
        parent_hash = _safe_str(record.get("parent_config_hash"), None)
        if not parent_hash or parent_hash not in row_map or parent_hash == config_hash:
            return config_hash
        return _root_for(parent_hash)

    @lru_cache(maxsize=None)
    def _descendants_for(config_hash: str) -> tuple[str, ...]:
        descendants: list[str] = []
        for child_hash in children_by_parent.get(config_hash, []):
            if child_hash not in descendants:
                descendants.append(child_hash)
            for nested_hash in _descendants_for(child_hash):
                if nested_hash not in descendants:
                    descendants.append(nested_hash)
        return tuple(descendants)

    def _lineage_record(config_hash: str, record: dict[str, Any]) -> dict[str, Any]:
        descendant_hashes = list(_descendants_for(config_hash))
        descendant_frame = canonical[canonical["config_hash"].isin(descendant_hashes)].copy() if descendant_hashes else pd.DataFrame()
        descendant_count = int(len(descendant_hashes))
        confirmation_descendant_count = 0
        holdout_descendant_count = 0
        rejected_descendant_count = 0
        invalid_descendant_count = 0
        no_trades_descendant_count = 0
        failed_confirmation_descendant_count = 0
        failed_holdout_descendant_count = 0
        targeted_follow_up_failure_count = 0
        branch_last_seen_timestamp_utc: str | None = None
        if not descendant_frame.empty:
            if "timestamp_utc" in descendant_frame.columns:
                ordered_descendants = descendant_frame.sort_values("timestamp_utc", kind="mergesort")
                branch_last_seen_timestamp_utc = _safe_str(ordered_descendants.iloc[-1].get("timestamp_utc"), None)
            for _, descendant in descendant_frame.iterrows():
                lineage_type = infer_lineage_type(descendant)
                if lineage_type == "confirmation":
                    confirmation_descendant_count += 1
                if lineage_type == "holdout" or _is_holdout_descendant(descendant):
                    holdout_descendant_count += 1
                status = str(descendant.get("status") or "").strip().lower()
                confirmation_outcome = str(descendant.get("confirmation_outcome") or "").strip().lower()
                holdout_outcome = str(descendant.get("holdout_check_outcome") or "").strip().lower()
                promotion_state = str(descendant.get("promotion_state") or "").strip().lower()
                status = str(descendant.get("status") or "").strip().lower()
                if (
                    status in _RESULT_STATUS_REJECTED
                    or confirmation_outcome in {"failed", "rejected"}
                    or holdout_outcome in {"failed", "rejected"}
                    or promotion_state == "rejected"
                ):
                    rejected_descendant_count += 1
                if status in {"invalid", "error"}:
                    invalid_descendant_count += 1
                if status == "no_trades":
                    no_trades_descendant_count += 1
                if confirmation_outcome in {"failed", "rejected"}:
                    failed_confirmation_descendant_count += 1
                if holdout_outcome in {"failed", "rejected"}:
                    failed_holdout_descendant_count += 1
                targeted_follow_up_required = _truthy(descendant.get("targeted_follow_up_required"))
                if targeted_follow_up_required and (confirmation_outcome in {"failed", "rejected"} or holdout_outcome in {"failed", "rejected"} or promotion_state == "rejected"):
                    targeted_follow_up_failure_count += 1
        best_descendant: dict[str, Any] | None = None
        if not descendant_frame.empty:
            ranked_descendants = _sort_descendants(descendant_frame)
            if not ranked_descendants.empty:
                best_descendant = _json_safe(ranked_descendants.iloc[0].to_dict())
        lineage_type = infer_lineage_type(pd.Series(record))
        parent_objective_score = _safe_float(record.get("objective_score"), 0.0) if record.get("objective_score") is not None else None
        best_descendant_objective_score = _safe_float(best_descendant.get("objective_score"), 0.0) if best_descendant else None
        lineage_root_config_hash = _root_for(config_hash)
        lineage_depth = 0
        parent_hash = _safe_str(record.get("parent_config_hash"), None)
        while parent_hash and parent_hash in row_map and parent_hash != config_hash:
            lineage_depth += 1
            parent_hash = _safe_str(row_map.get(parent_hash, {}).get("parent_config_hash"), None)
            if lineage_depth > len(row_map):
                break
        lineage_status_summary = _lineage_status_summary(
            lineage_type=lineage_type,
            descendant_count=descendant_count,
            confirmation_descendant_count=confirmation_descendant_count,
            holdout_descendant_count=holdout_descendant_count,
            rejected_descendant_count=rejected_descendant_count,
            parent_objective_score=parent_objective_score,
            best_descendant_objective_score=best_descendant_objective_score,
        )
        lineage_trust_score = _lineage_trust_score(
            descendant_count=descendant_count,
            confirmation_descendant_count=confirmation_descendant_count,
            holdout_descendant_count=holdout_descendant_count,
            rejected_descendant_count=rejected_descendant_count,
            parent_objective_score=parent_objective_score,
            best_descendant_objective_score=best_descendant_objective_score,
            lineage_type=lineage_type,
        )
        branch_balance = _branch_balance(
            descendant_count=descendant_count,
            confirmation_descendant_count=confirmation_descendant_count,
            holdout_descendant_count=holdout_descendant_count,
            rejected_descendant_count=rejected_descendant_count,
        )
        branch_age_days = _days_between(branch_last_seen_timestamp_utc or record.get("timestamp_utc"), latest_batch.get("summary", {}).get("timestamp_utc") if latest_batch else datetime.now(UTC).isoformat())
        persisted_record = (persisted_records or {}).get(str(record.get("strategy_family") or ""), {}).get(config_hash)
        previous_branch_state = _safe_str((persisted_record or {}).get("branch_state"), None)
        previous_branch_decay_score = _safe_float((persisted_record or {}).get("branch_decay_score"), None)
        previous_best_descendant_objective_score = _safe_float(
            (persisted_record or {}).get("best_descendant_objective_score") or (persisted_record or {}).get("best_branch_objective_score"),
            None,
        )
        decay_score, branch_state, _branch_status_summary, branch_retirement_reason, recovery_signal, branch_revival_condition = _branch_decay_state(
            lineage_type=lineage_type,
            lineage_status_summary=lineage_status_summary,
            lineage_trust_score=lineage_trust_score,
            branch_balance=branch_balance,
            descendant_count=descendant_count,
            confirmation_descendant_count=confirmation_descendant_count,
            holdout_descendant_count=holdout_descendant_count,
            rejected_descendant_count=rejected_descendant_count,
            invalid_descendant_count=invalid_descendant_count,
            no_trades_descendant_count=no_trades_descendant_count,
            failed_confirmation_descendant_count=failed_confirmation_descendant_count,
            failed_holdout_descendant_count=failed_holdout_descendant_count,
            targeted_follow_up_failure_count=targeted_follow_up_failure_count,
            branch_age_days=branch_age_days,
            best_descendant_objective_score=best_descendant_objective_score,
            parent_objective_score=parent_objective_score,
            previous_branch_state=previous_branch_state,
            previous_branch_decay_score=previous_branch_decay_score,
            previous_best_descendant_objective_score=previous_best_descendant_objective_score,
        )
        if include_histories:
            merged_history = list(record.get("history") or record.get("lineage_history") or [])
            if isinstance(persisted_record, dict):
                for item in persisted_record.get("history") or persisted_record.get("lineage_history") or []:
                    if isinstance(item, dict) and item not in merged_history:
                        merged_history.append(dict(item))
            current_event = {
                "timestamp_utc": record.get("timestamp_utc"),
                "experiment_id": record.get("experiment_id"),
                "lineage_status_summary": lineage_status_summary,
                "lineage_trust_score": round(lineage_trust_score, 6),
                "branch_state": branch_state,
                "branch_decay_score": round(decay_score, 6),
                "descendant_count": descendant_count,
                "confirmation_descendant_count": confirmation_descendant_count,
                "holdout_descendant_count": holdout_descendant_count,
                "rejected_descendant_count": rejected_descendant_count,
            }
            if current_event not in merged_history:
                merged_history.append(current_event)
            merged_history = merged_history[-50:]
        else:
            merged_history = []
        return {
            "family": record.get("strategy_family"),
            "config_hash": config_hash,
            "experiment_id": record.get("experiment_id"),
            "parent_config_hash": record.get("parent_config_hash"),
            "lineage_root_config_hash": lineage_root_config_hash,
            "lineage_depth": lineage_depth,
            "lineage_type": lineage_type,
            "descendant_count": descendant_count,
            "confirmation_descendant_count": confirmation_descendant_count,
            "holdout_descendant_count": holdout_descendant_count,
            "rejected_descendant_count": rejected_descendant_count,
            "best_descendant_config_hash": best_descendant.get("config_hash") if best_descendant else None,
            "best_descendant_experiment_id": best_descendant.get("experiment_id") if best_descendant else None,
            "best_descendant_objective_score": best_descendant.get("objective_score") if best_descendant else None,
            "best_descendant_status": best_descendant.get("status") if best_descendant else None,
            "lineage_status_summary": lineage_status_summary,
            "lineage_trust_score": round(lineage_trust_score, 6),
            "branch_balance": round(branch_balance, 6),
            "branch_decay_score": round(decay_score, 6),
            "branch_state": branch_state,
            "branch_retirement_reason": branch_retirement_reason,
            "branch_revival_condition": branch_revival_condition,
            "branch_age_days": branch_age_days,
            "branch_last_seen_timestamp_utc": branch_last_seen_timestamp_utc,
            "invalid_descendant_count": invalid_descendant_count,
            "no_trades_descendant_count": no_trades_descendant_count,
            "failed_confirmation_descendant_count": failed_confirmation_descendant_count,
            "failed_holdout_descendant_count": failed_holdout_descendant_count,
            "targeted_follow_up_failure_count": targeted_follow_up_failure_count,
            "branch_recovery_signal": recovery_signal,
            "source_batch_ids": _json_safe(record.get("source_batch_ids") or []),
            "source_proposal_id": record.get("source_proposal_id"),
            "history": merged_history if include_histories else [],
            "lineage_history": merged_history if include_histories else [],
            "last_seen_timestamp_utc": record.get("timestamp_utc"),
            "last_seen_cycle_id": record.get("experiment_id"),
            "updated_at": latest_batch.get("summary", {}).get("timestamp_utc") if latest_batch else record.get("timestamp_utc"),
        }

    records_by_hash: dict[str, dict[str, Any]] = {}
    records_by_family: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    lineage_status_counts: Counter[str] = Counter()
    for config_hash, record in row_map.items():
        lineage_record = _lineage_record(config_hash, record)
        records_by_hash[config_hash] = lineage_record
        records_by_family[str(record.get("strategy_family") or "").strip().lower()][config_hash] = lineage_record
        lineage_status_counts[lineage_record["lineage_status_summary"]] += 1

    family_summaries: dict[str, dict[str, Any]] = {}
    branch_summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for family, family_records in records_by_family.items():
        if not family_records:
            continue
        ranked = sorted(
            family_records.values(),
            key=lambda record: (
                {"active": 4, "revivable": 3, "decaying": 2, "paused": 1, "retired": 0}.get(str(record.get("branch_state") or "active"), 0),
                -float(record.get("branch_decay_score") or 0.0),
                -float(record.get("lineage_trust_score") or 0.0),
                -_sort_numeric(record.get("best_descendant_objective_score")),
                -int(record.get("descendant_count") or 0),
                str(record.get("config_hash") or ""),
            ),
        )
        top_record = dict(ranked[0])
        family_summaries[family] = {
            "family": family,
            "result_count": len(family_records),
            "root_count": int(
                len({str(record.get("lineage_root_config_hash") or record.get("config_hash") or "") for record in family_records.values()})
            ),
            "strengthening_count": int(sum(1 for record in family_records.values() if record.get("lineage_status_summary") == "strengthening")),
            "mixed_count": int(sum(1 for record in family_records.values() if record.get("lineage_status_summary") == "mixed")),
            "failing_count": int(sum(1 for record in family_records.values() if record.get("lineage_status_summary") == "failing")),
            "leaf_count": int(sum(1 for record in family_records.values() if record.get("lineage_status_summary") == "leaf")),
            "lineage_status_summary": top_record.get("lineage_status_summary"),
            "lineage_trust_score": top_record.get("lineage_trust_score"),
            "lineage_root_config_hash": top_record.get("lineage_root_config_hash"),
            "lineage_type": top_record.get("lineage_type"),
            "lineage_depth": top_record.get("lineage_depth"),
            "descendant_count": top_record.get("descendant_count"),
            "confirmation_descendant_count": top_record.get("confirmation_descendant_count"),
            "holdout_descendant_count": top_record.get("holdout_descendant_count"),
            "rejected_descendant_count": top_record.get("rejected_descendant_count"),
            "best_descendant_config_hash": top_record.get("best_descendant_config_hash"),
            "best_descendant_objective_score": top_record.get("best_descendant_objective_score"),
            "branch_balance": top_record.get("branch_balance"),
            "branch_decay_score": top_record.get("branch_decay_score"),
            "branch_state": top_record.get("branch_state"),
            "branch_retirement_reason": top_record.get("branch_retirement_reason"),
            "branch_revival_condition": top_record.get("branch_revival_condition"),
            "branch_age_days": top_record.get("branch_age_days"),
            "branch_last_seen_timestamp_utc": top_record.get("branch_last_seen_timestamp_utc"),
            "invalid_descendant_count": top_record.get("invalid_descendant_count"),
            "no_trades_descendant_count": top_record.get("no_trades_descendant_count"),
            "failed_confirmation_descendant_count": top_record.get("failed_confirmation_descendant_count"),
            "failed_holdout_descendant_count": top_record.get("failed_holdout_descendant_count"),
            "targeted_follow_up_failure_count": top_record.get("targeted_follow_up_failure_count"),
            "branch_recovery_signal": top_record.get("branch_recovery_signal"),
            "history": top_record.get("history") or [] if include_histories else [],
            "lineage_history": top_record.get("lineage_history") or top_record.get("history") or [] if include_histories else [],
        }

        branch_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in family_records.values():
            branch_root = str(record.get("lineage_root_config_hash") or record.get("config_hash") or "").strip()
            if branch_root:
                branch_groups[branch_root].append(record)
        for branch_root, branch_records in branch_groups.items():
            if not branch_records:
                continue
            ranked_branch = sorted(
                branch_records,
                key=lambda record: (
                    -float(record.get("lineage_trust_score") or 0.0),
                    -_sort_numeric(record.get("best_descendant_objective_score")),
                    -int(record.get("descendant_count") or 0),
                    str(record.get("config_hash") or ""),
                ),
            )
            branch_root_record = next(
                (record for record in branch_records if str(record.get("config_hash") or "") == branch_root),
                ranked_branch[0],
            )
            branch_best = ranked_branch[0]
            branch_trust = float(branch_root_record.get("lineage_trust_score") or branch_best.get("lineage_trust_score") or 0.0)
            branch_summaries[family][branch_root] = {
                "family": family,
                "branch_root_config_hash": branch_root,
                "branch_root_experiment_id": branch_root_record.get("experiment_id"),
                "branch_root_status": branch_root_record.get("status"),
                "branch_root_lineage_type": branch_root_record.get("lineage_type"),
                "branch_status_summary": branch_root_record.get("lineage_status_summary"),
                "branch_trust_score": round(branch_trust, 6),
                "branch_balance": branch_root_record.get("branch_balance"),
                "branch_decay_score": branch_root_record.get("branch_decay_score"),
                "branch_state": branch_root_record.get("branch_state"),
                "branch_retirement_reason": branch_root_record.get("branch_retirement_reason"),
                "branch_revival_condition": branch_root_record.get("branch_revival_condition"),
                "branch_age_days": branch_root_record.get("branch_age_days"),
                "branch_last_seen_timestamp_utc": branch_root_record.get("branch_last_seen_timestamp_utc"),
                "invalid_descendant_count": branch_root_record.get("invalid_descendant_count"),
                "no_trades_descendant_count": branch_root_record.get("no_trades_descendant_count"),
                "failed_confirmation_descendant_count": branch_root_record.get("failed_confirmation_descendant_count"),
                "failed_holdout_descendant_count": branch_root_record.get("failed_holdout_descendant_count"),
                "targeted_follow_up_failure_count": branch_root_record.get("targeted_follow_up_failure_count"),
                "branch_recovery_signal": branch_root_record.get("branch_recovery_signal"),
                "branch_depth": branch_root_record.get("lineage_depth"),
                "descendant_count": branch_root_record.get("descendant_count"),
                "confirmation_descendant_count": branch_root_record.get("confirmation_descendant_count"),
                "holdout_descendant_count": branch_root_record.get("holdout_descendant_count"),
                "rejected_descendant_count": branch_root_record.get("rejected_descendant_count"),
                "best_branch_config_hash": branch_best.get("config_hash"),
                "best_branch_experiment_id": branch_best.get("experiment_id"),
                "best_branch_objective_score": branch_best.get("objective_score"),
                "best_branch_status": branch_best.get("status"),
                "branch_node_count": int(len(branch_records)),
                "branch_history": branch_root_record.get("lineage_history") or branch_root_record.get("history") or [] if include_histories else [],
                "history": branch_root_record.get("history") or [] if include_histories else [],
                "updated_at": branch_root_record.get("updated_at"),
            }

    return {
        "result_count": int(len(canonical)),
        "latest_batch_id": latest_batch.get("batch_id") if latest_batch else None,
        "records": records_by_family if include_records else {},
        "by_config_hash": records_by_hash,
        "family_summaries": family_summaries,
        "branch_summaries": branch_summaries,
        "lineage_status_counts": dict(lineage_status_counts),
    }


def _branch_score(branch: dict[str, Any], *, cycle_mode: str, confirmation_family: str | None) -> tuple[float, list[str]]:
    trust = _safe_float(branch.get("branch_trust_score"), 0.0)
    balance = _safe_float(branch.get("branch_balance"), 0.0)
    descendant_count = int(branch.get("descendant_count") or 0)
    confirmation_descendants = int(branch.get("confirmation_descendant_count") or 0)
    holdout_descendants = int(branch.get("holdout_descendant_count") or 0)
    rejected_descendants = int(branch.get("rejected_descendant_count") or 0)
    status = str(branch.get("branch_status_summary") or "leaf").strip().lower() or "leaf"
    branch_state = str(branch.get("branch_state") or "active").strip().lower() or "active"
    decay_score = _safe_float(branch.get("branch_decay_score"), 0.0)
    best_objective = _safe_float(branch.get("best_branch_objective_score"), float("-inf"))
    root_hash = _safe_str(branch.get("branch_root_config_hash"), None)
    state_multiplier = {
        "active": 1.0,
        "revivable": 0.65,
        "decaying": 0.45,
        "paused": 0.08,
        "retired": 0.0,
    }.get(branch_state, 0.3)
    score = (0.20 + (0.45 * trust) + (0.15 * max(0.0, balance)) + (0.10 * min(1.0, (confirmation_descendants + holdout_descendants) / max(1, descendant_count)))) * state_multiplier
    score -= 0.10 * decay_score
    reasons: list[str] = []
    if status == "strengthening":
        score += 0.18
        reasons.append("branch_strengthening")
    elif status == "mixed":
        score += 0.03
        reasons.append("branch_mixed")
    elif status == "leaf":
        score -= 0.05
        reasons.append("branch_leaf")
    elif status == "failing":
        score -= 0.28
        reasons.append("branch_failing")
    if rejected_descendants >= max(2, confirmation_descendants + holdout_descendants):
        score -= 0.18
        reasons.append("branch_rejected_heavy")
    if best_objective > float("-inf"):
        score += min(0.10, max(-0.05, best_objective * 0.03))
        reasons.append("branch_objective_signal")
    if cycle_mode == "confirmation" and confirmation_family and branch.get("family") == confirmation_family:
        score += 0.15
        reasons.append("confirmation_focus")
    if cycle_mode == "diagnostics" and status in {"failing", "mixed"}:
        score += 0.05
        reasons.append("diagnostics_focus")
    if cycle_mode == "stagnation_escape" and status == "strengthening":
        score += 0.08
        reasons.append("stagnation_escape_strengthening")
    if branch_state == "revivable":
        score += 0.05
        reasons.append("branch_revivable")
    elif branch_state == "paused":
        score -= 0.08
        reasons.append("branch_paused")
    elif branch_state == "retired":
        score -= 0.20
        reasons.append("branch_retired")
    elif branch_state == "decaying":
        score -= 0.05
        reasons.append("branch_decaying")
    if branch.get("branch_recovery_signal"):
        score += 0.08
        reasons.append("branch_recovery_signal")
    return max(score, 0.0), [reason for reason in reasons if reason] + ([f"branch_root={root_hash}"] if root_hash else [])


def build_branch_budget_plan(
    lineage_summary: dict[str, Any] | None,
    family_budgets: dict[str, int] | None,
    *,
    cycle_mode: str = "normal_exploration",
    confirmation_family: str | None = None,
    confirmation_required: bool = False,
    targeted_follow_up_required: bool = False,
    holdout_check_required: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    lineage_summary = lineage_summary or {}
    family_budgets = dict(family_budgets or {})
    branch_summary_lookup = lineage_summary.get("branch_summaries") or {}
    branch_budgets: dict[str, list[dict[str, Any]]] = {}
    rationale: dict[str, Any] = {
        "cycle_mode": cycle_mode,
        "confirmation_family": confirmation_family,
        "confirmation_required": confirmation_required,
        "targeted_follow_up_required": targeted_follow_up_required,
        "holdout_check_required": holdout_check_required,
        "families": {},
    }

    for family, family_budget in family_budgets.items():
        branch_records = dict(branch_summary_lookup.get(family) or {})
        if family_budget <= 0:
            branch_budgets[family] = []
            rationale["families"][family] = {
                "family_budget": 0,
                "reason": "family budget is zero, so no branch allocation was assigned",
                "branches": [],
            }
            continue
        if not branch_records:
            branch_budgets[family] = []
            rationale["families"][family] = {
                "family_budget": family_budget,
                "reason": "no lineage branches available; branch-level allocation deferred to candidate generation fallback",
                "branches": [],
            }
            continue

        scored: list[dict[str, Any]] = []
        for branch in branch_records.values():
            branch_score, reasons = _branch_score(
                branch,
                cycle_mode=cycle_mode,
                confirmation_family=confirmation_family,
            )
            branch = dict(branch)
            branch["branch_score"] = round(branch_score, 6)
            branch["branch_score_reasons"] = reasons
            scored.append(branch)
        scored.sort(
            key=lambda record: (
                -float(record.get("branch_score") or 0.0),
                -float(record.get("branch_trust_score") or 0.0),
                -int(record.get("descendant_count") or 0),
                str(record.get("branch_root_config_hash") or ""),
            )
        )
        effective_scored = [
            dict(branch, branch_effective_score=round(float(branch.get("branch_score") or 0.0) * _branch_state_multiplier(str(branch.get("branch_state") or "active")), 6))
            for branch in scored
        ]
        if confirmation_required or targeted_follow_up_required or holdout_check_required:
            eligible = [branch for branch in effective_scored if str(branch.get("branch_state") or "").strip().lower() not in {"paused", "retired"}]
            if not eligible:
                branch_budgets[family] = []
                rationale["families"][family] = {
                    "family_budget": family_budget,
                    "reason": "all lineage branches are paused or retired, so branch-level allocation is withheld until new evidence appears",
                    "branches": scored,
                }
                continue
            top_branch = dict(eligible[0])
            top_branch["branch_budget"] = family_budget
            top_branch["branch_budget_share"] = 1.0
            top_branch["branch_budget_stance"] = "primary"
            top_branch["branch_budget_reason"] = "confirmation or targeted validation is focusing the family budget on the strongest lineage branch"
            top_branch["branch_budget_rationale"] = top_branch["branch_budget_reason"]
            for branch in scored:
                if branch.get("branch_root_config_hash") == top_branch.get("branch_root_config_hash"):
                    continue
                branch.update(
                    {
                        "branch_budget": 0,
                        "branch_budget_share": 0.0,
                        "branch_budget_stance": "paused",
                        "branch_budget_reason": "branch held back during confirmation or targeted validation",
                        "branch_budget_rationale": "branch held back during confirmation or targeted validation",
                    }
                )
            branch_budgets[family] = [top_branch] + [branch for branch in scored if branch.get("branch_root_config_hash") != top_branch.get("branch_root_config_hash")]
            rationale["families"][family] = {
                "family_budget": family_budget,
                "reason": "confirmation or holdout validation concentrates budget on the most trustworthy branch",
                "branches": branch_budgets[family],
            }
            continue

        active_effective_scores = [
            max(float(branch.get("branch_effective_score") or 0.0), 0.01)
            for branch in effective_scored
            if str(branch.get("branch_state") or "").strip().lower() not in {"paused", "retired"}
        ]
        total_score = sum(active_effective_scores)
        if total_score <= 0:
            total_score = float(len(active_effective_scores) or len(effective_scored) or 1)
        allocated: list[dict[str, Any]] = []
        for branch in effective_scored:
            branch_state = str(branch.get("branch_state") or "active").strip().lower()
            if branch_state in {"paused", "retired"}:
                score = 0.0
            else:
                score = max(float(branch.get("branch_effective_score") or branch.get("branch_score") or 0.0), 0.01)
            share = score / total_score
            state_multiplier = {
                "active": 1.0,
                "revivable": 0.60,
                "decaying": 0.35,
                "paused": 0.0,
                "retired": 0.0,
            }.get(branch_state, 0.25)
            budget = int(round(family_budget * share * state_multiplier))
            if branch_state in {"paused", "retired"}:
                budget = 0
            branch = dict(branch)
            branch["branch_budget_share"] = round(share, 6)
            branch["branch_budget"] = max(0, budget)
            if branch_state == "active":
                branch["branch_budget_stance"] = "primary" if share >= 0.40 and budget > 0 else "controlled" if budget > 0 else "paused"
                branch["branch_budget_reason"] = (
                    "branch remains active and supported by lineage evidence"
                    if branch["branch_budget_stance"] == "primary"
                    else "branch retained with controlled budget because decay is still moderate"
                    if branch["branch_budget_stance"] == "controlled"
                    else "branch budget withheld because lineage evidence is too weak"
                )
                branch["branch_budget_rationale"] = branch["branch_budget_reason"]
            elif branch_state == "revivable":
                branch["branch_budget_stance"] = "controlled" if budget > 0 else "paused"
                branch["branch_budget_reason"] = (
                    "branch has meaningful recovery evidence, so it keeps a small controlled budget"
                    if branch["branch_budget_stance"] == "controlled"
                    else "branch revival is not yet strong enough for budget"
                )
                branch["branch_budget_rationale"] = branch["branch_budget_reason"]
            elif branch_state == "decaying":
                branch["branch_budget_stance"] = "controlled" if budget > 0 else "paused"
                branch["branch_budget_reason"] = (
                    "branch is decaying, so its budget is reduced"
                    if branch["branch_budget_stance"] == "controlled"
                    else "branch is decaying too sharply to justify budget"
                )
                branch["branch_budget_rationale"] = branch["branch_budget_reason"]
            elif branch_state == "paused":
                branch["branch_budget_stance"] = "paused"
                branch["branch_budget_reason"] = "branch is paused because repeated failures and staleness outweigh support"
                branch["branch_budget_rationale"] = branch["branch_budget_reason"]
            else:
                branch["branch_budget_stance"] = "retired"
                branch["branch_budget_reason"] = "branch is retired because repeated failures, stale evidence, and low trust dominate"
                branch["branch_budget_rationale"] = branch["branch_budget_reason"]
            allocated.append(branch)
        # Ensure the branch budget sums to the family budget while keeping a primary branch alive.
        total_allocated = sum(int(branch.get("branch_budget") or 0) for branch in allocated)
        while total_allocated < family_budget and allocated:
            primary = max(
                allocated,
                key=lambda record: (
                    float(record.get("branch_budget") or 0),
                    float(record.get("branch_score") or 0.0),
                    str(record.get("branch_root_config_hash") or ""),
                ),
            )
            primary["branch_budget"] = int(primary.get("branch_budget") or 0) + 1
            total_allocated += 1
        while total_allocated > family_budget and allocated:
            donor_candidates = [branch for branch in allocated if int(branch.get("branch_budget") or 0) > 0]
            if not donor_candidates:
                break
            donor = min(
                donor_candidates,
                key=lambda record: (
                    float(record.get("branch_score") or 0.0),
                    float(record.get("branch_budget") or 0.0),
                    str(record.get("branch_root_config_hash") or ""),
                ),
            )
            donor["branch_budget"] = int(donor.get("branch_budget") or 0) - 1
            total_allocated -= 1
        branch_budgets[family] = allocated
        rationale["families"][family] = {
            "family_budget": family_budget,
            "reason": "budget allocated across lineage branches using trust, descendants, and branch status",
            "branches": allocated,
        }

    return branch_budgets, rationale


def annotate_lineage(frame: pd.DataFrame, lineage_summary: dict[str, Any] | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    annotated = frame.copy()
    columns = (
        "lineage_root_config_hash",
        "lineage_depth",
        "lineage_type",
        "descendant_count",
        "confirmation_descendant_count",
        "holdout_descendant_count",
        "rejected_descendant_count",
        "best_descendant_config_hash",
        "best_descendant_experiment_id",
        "best_descendant_objective_score",
        "best_descendant_status",
        "lineage_status_summary",
        "lineage_trust_score",
        "branch_balance",
        "branch_decay_score",
        "branch_state",
        "branch_retirement_reason",
        "branch_revival_condition",
        "branch_age_days",
        "branch_last_seen_timestamp_utc",
        "invalid_descendant_count",
        "no_trades_descendant_count",
        "failed_confirmation_descendant_count",
        "failed_holdout_descendant_count",
        "targeted_follow_up_failure_count",
        "branch_recovery_signal",
        "lineage_history",
    )
    for column in columns:
        if column not in annotated.columns:
            annotated[column] = None
    if not lineage_summary:
        return annotated
    lookup = lineage_summary.get("by_config_hash") or {}
    if "config_hash" not in annotated.columns:
        return annotated
    for idx, row in annotated.iterrows():
        config_hash = _safe_str(row.get("config_hash"), None)
        if not config_hash or config_hash not in lookup:
            continue
        record = lookup.get(config_hash) or {}
        for column in columns:
            if column in record:
                annotated.at[idx, column] = record.get(column)
    return annotated
