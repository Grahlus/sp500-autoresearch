from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiment_types import PromotionStateRecord


MEMORY_FILE = "memory.json"
PROMOTION_STATES_KEY = "promotion_states"
LINEAGE_STATES_KEY = "lineage_states"
IDEA_YIELD_STATES_KEY = "idea_yield_states"
PROMOTION_HISTORY_LIMIT = 50


def _memory_path(base_dir: str) -> Path:
    return Path(base_dir) / MEMORY_FILE


def load_research_memory(base_dir: str = "experiments") -> dict[str, Any]:
    path = _memory_path(base_dir)
    if not path.exists():
        return {
            "version": 1,
            "updated_at": None,
            "families": {},
            PROMOTION_STATES_KEY: {},
            LINEAGE_STATES_KEY: {},
            IDEA_YIELD_STATES_KEY: {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return {
            "version": 1,
            "updated_at": None,
            "families": {},
            PROMOTION_STATES_KEY: {},
            LINEAGE_STATES_KEY: {},
        }
    payload.setdefault("families", {})
    payload.setdefault(PROMOTION_STATES_KEY, {})
    payload.setdefault(LINEAGE_STATES_KEY, {})
    payload.setdefault(IDEA_YIELD_STATES_KEY, {})
    return payload


def save_research_memory(memory: dict[str, Any], base_dir: str = "experiments") -> None:
    path = _memory_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(memory)
    payload["updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def update_family_memory(
    memory: dict[str, Any],
    family: str,
    *,
    exact_hashes: set[str],
    coarse_signatures: set[str],
    dead_zone_values: dict[str, set[Any]],
    dead_zone_signatures: set[str],
    poor_region_signatures: set[str] | None = None,
    template_counts: dict[str, int],
    best_objective_score: float | None,
    best_config_hash: str | None,
    stagnation_batches: int,
) -> dict[str, Any]:
    families = memory.setdefault("families", {})
    families[family] = {
        "exact_hashes": sorted(exact_hashes),
        "coarse_signatures": sorted(coarse_signatures),
        "dead_zone_values": {key: sorted(str(value) for value in values) for key, values in dead_zone_values.items()},
        "dead_zone_signatures": sorted(dead_zone_signatures),
        "poor_region_signatures": sorted(poor_region_signatures or set()),
        "template_counts": dict(sorted(template_counts.items())),
        "best_objective_score": best_objective_score,
        "best_config_hash": best_config_hash,
        "stagnation_batches": int(stagnation_batches),
    }
    return memory


def _history_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (str(entry.get("decision_id") or entry.get("cycle_id") or entry.get("timestamp_utc") or ""), str(entry.get("event_type") or ""))


def _merge_history(existing: list[dict[str, Any]] | None, new: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = [dict(item) for item in (existing or []) if isinstance(item, dict)]
    seen = {(_history_key(item)) for item in merged}
    for item in new or []:
        if not isinstance(item, dict):
            continue
        key = _history_key(item)
        if key in seen:
            for idx, existing_item in enumerate(merged):
                if _history_key(existing_item) == key:
                    merged[idx] = {**existing_item, **item}
                    break
            continue
        merged.append(dict(item))
        seen.add(key)
    if len(merged) > PROMOTION_HISTORY_LIMIT:
        merged = merged[-PROMOTION_HISTORY_LIMIT:]
    return merged


def get_promotion_state_record(memory: dict[str, Any], family: str, config_hash: str) -> dict[str, Any] | None:
    promotion_states = memory.get(PROMOTION_STATES_KEY) or {}
    family_bucket = promotion_states.get(str(family), {}) if isinstance(promotion_states, dict) else {}
    record = family_bucket.get(str(config_hash))
    if isinstance(record, dict):
        return dict(record)
    return None


def get_lineage_state_record(memory: dict[str, Any], family: str, config_hash: str) -> dict[str, Any] | None:
    lineage_states = memory.get(LINEAGE_STATES_KEY) or {}
    family_bucket = lineage_states.get(str(family), {}) if isinstance(lineage_states, dict) else {}
    record = family_bucket.get(str(config_hash))
    if isinstance(record, dict):
        return dict(record)
    return None


def load_promotion_state_records(base_dir: str = "experiments") -> dict[str, dict[str, dict[str, Any]]]:
    memory = load_research_memory(base_dir)
    promotion_states = memory.get(PROMOTION_STATES_KEY) or {}
    if not isinstance(promotion_states, dict):
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family, configs in promotion_states.items():
        if not isinstance(configs, dict):
            continue
        result[str(family)] = {str(config_hash): dict(record) for config_hash, record in configs.items() if isinstance(record, dict)}
    return result


def load_lineage_state_records(base_dir: str = "experiments") -> dict[str, dict[str, dict[str, Any]]]:
    memory = load_research_memory(base_dir)
    lineage_states = memory.get(LINEAGE_STATES_KEY) or {}
    if not isinstance(lineage_states, dict):
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family, configs in lineage_states.items():
        if not isinstance(configs, dict):
            continue
        result[str(family)] = {str(config_hash): dict(record) for config_hash, record in configs.items() if isinstance(record, dict)}
    return result


def update_promotion_state_record(
    memory: dict[str, Any],
    record: PromotionStateRecord | dict[str, Any],
) -> dict[str, Any]:
    payload = dict(record if isinstance(record, dict) else record.__dict__)
    family = str(payload.get("family") or "").strip().lower()
    config_hash = str(payload.get("config_hash") or "").strip()
    if not family or not config_hash:
        return memory

    promotion_states = memory.setdefault(PROMOTION_STATES_KEY, {})
    family_bucket = promotion_states.setdefault(family, {})
    existing = dict(family_bucket.get(config_hash) or {})
    merged = {**existing, **payload}
    merged["family"] = family
    merged["config_hash"] = config_hash
    merged["promotion_state"] = str(merged.get("promotion_state") or "unconfirmed")
    merged["winner_promotion_status"] = str(merged.get("winner_promotion_status") or "not_promoted")
    merged["confirmation_required"] = bool(merged.get("confirmation_required"))
    merged["holdout_check_required"] = bool(merged.get("holdout_check_required"))
    merged["blocked_pending_new_evidence"] = bool(merged.get("blocked_pending_new_evidence"))
    merged["confirmation_history"] = _merge_history(existing.get("confirmation_history"), payload.get("confirmation_history"))
    merged["holdout_history"] = _merge_history(existing.get("holdout_history"), payload.get("holdout_history"))
    merged["history"] = _merge_history(existing.get("history"), payload.get("history"))
    for key in ("source_batch_ids", "validation_horizon_tags", "validation_regime_tags"):
        value = merged.get(key)
        if value is None:
            merged[key] = []
        elif not isinstance(value, list):
            merged[key] = list(value) if isinstance(value, tuple) else [value]
    merged["updated_at"] = payload.get("updated_at") or datetime.now(UTC).isoformat()
    family_bucket[config_hash] = merged
    return memory


def update_lineage_state_record(
    memory: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    family = str(record.get("family") or "").strip().lower()
    config_hash = str(record.get("config_hash") or "").strip()
    if not family or not config_hash:
        return memory

    lineage_states = memory.setdefault(LINEAGE_STATES_KEY, {})
    family_bucket = lineage_states.setdefault(family, {})
    existing = dict(family_bucket.get(config_hash) or {})
    merged = {**existing, **dict(record)}
    merged["family"] = family
    merged["config_hash"] = config_hash
    merged["lineage_type"] = str(merged.get("lineage_type") or "seed")
    merged["lineage_status_summary"] = str(merged.get("lineage_status_summary") or "seed")
    merged["descendant_count"] = int(merged.get("descendant_count") or 0)
    merged["confirmation_descendant_count"] = int(merged.get("confirmation_descendant_count") or 0)
    merged["holdout_descendant_count"] = int(merged.get("holdout_descendant_count") or 0)
    merged["rejected_descendant_count"] = int(merged.get("rejected_descendant_count") or 0)
    merged["lineage_trust_score"] = float(merged.get("lineage_trust_score") or 0.0)
    merged["branch_balance"] = float(merged.get("branch_balance") or 0.0)
    history = _merge_history(existing.get("history"), merged.get("history"))
    merged["history"] = history
    merged["lineage_history"] = history
    for key in ("source_batch_ids",):
        value = merged.get(key)
        if value is None:
            merged[key] = []
        elif not isinstance(value, list):
            merged[key] = list(value) if isinstance(value, tuple) else [value]
    merged["updated_at"] = merged.get("updated_at") or datetime.now(UTC).isoformat()
    family_bucket[config_hash] = merged
    return memory


def get_idea_yield_state_record(memory: dict[str, Any], family: str, idea_key: str) -> dict[str, Any] | None:
    idea_yield_states = memory.get(IDEA_YIELD_STATES_KEY) or {}
    family_bucket = idea_yield_states.get(str(family), {}) if isinstance(idea_yield_states, dict) else {}
    record = family_bucket.get(str(idea_key))
    if isinstance(record, dict):
        return dict(record)
    return None


def load_idea_yield_state_records(base_dir: str = "experiments") -> dict[str, dict[str, dict[str, Any]]]:
    memory = load_research_memory(base_dir)
    idea_yield_states = memory.get(IDEA_YIELD_STATES_KEY) or {}
    if not isinstance(idea_yield_states, dict):
        return {}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family, records in idea_yield_states.items():
        if not isinstance(records, dict):
            continue
        result[str(family)] = {str(key): dict(record) for key, record in records.items() if isinstance(record, dict)}
    return result


def update_idea_yield_state_record(
    memory: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    family = str(record.get("family") or "").strip().lower()
    idea_key = str(record.get("idea_key") or "").strip()
    if not family or not idea_key:
        return memory
    idea_yield_states = memory.setdefault(IDEA_YIELD_STATES_KEY, {})
    family_bucket = idea_yield_states.setdefault(family, {})
    existing = dict(family_bucket.get(idea_key) or {})
    merged = {**existing, **dict(record)}
    merged["family"] = family
    merged["idea_key"] = idea_key
    merged["idea_kind"] = str(merged.get("idea_kind") or "unknown")
    merged["template_id"] = str(merged.get("template_id") or "unspecified")
    merged["strategy_type"] = str(merged.get("strategy_type") or "classical")
    merged["idea_source"] = str(merged.get("idea_source") or "unknown")
    merged["idea_state"] = str(merged.get("idea_state") or "untested")
    merged["idea_attempt_count"] = int(merged.get("idea_attempt_count") or 0)
    merged["idea_viable_count"] = int(merged.get("idea_viable_count") or 0)
    merged["idea_baseline_beat_count"] = int(merged.get("idea_baseline_beat_count") or 0)
    merged["idea_descendant_count"] = int(merged.get("idea_descendant_count") or 0)
    merged["idea_robust_descendant_count"] = int(merged.get("idea_robust_descendant_count") or 0)
    merged["idea_dead_zone_count"] = int(merged.get("idea_dead_zone_count") or 0)
    merged["idea_repeated_failure_count"] = int(merged.get("idea_repeated_failure_count") or 0)
    merged["updated_at"] = merged.get("updated_at") or datetime.now(UTC).isoformat()
    family_bucket[idea_key] = merged
    return memory


def load_promotion_state_records_for_family(base_dir: str, family: str) -> dict[str, dict[str, Any]]:
    records = load_promotion_state_records(base_dir)
    return dict(records.get(str(family), {}) or {})


def load_lineage_state_records_for_family(base_dir: str, family: str) -> dict[str, dict[str, Any]]:
    records = load_lineage_state_records(base_dir)
    return dict(records.get(str(family), {}) or {})
