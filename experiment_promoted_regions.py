from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from experiment_spaces import get_family_search_space, normalize_experiment_config
from experiment_store import compute_config_hash, load_results_index


PROMOTED_REGIONS_DIR = Path("reports/promoted_regions")
PROMOTED_REGIONS_INDEX = PROMOTED_REGIONS_DIR / "index.json"
PROMOTED_REGIONS_REPORT = PROMOTED_REGIONS_DIR / "report.json"

DEFAULT_REGION_FOLLOWUP_BUDGET = 4
DEFAULT_MAX_REGION_FOLLOWUP_CANDIDATES = 6
DEFAULT_MAX_ACTIVE_PROMOTED_REGIONS = 2
DEFAULT_REGION_FOLLOWUP_REPEAT_CAP = 1
DEFAULT_VALIDATED_REGION_NEIGHBOR_BUDGET = 2
DEFAULT_VALIDATED_REGION_NEIGHBOR_RADIUS = 1
DEFAULT_MAX_VALIDATED_REGION_NEIGHBORS = 4
DEFAULT_VALIDATED_REGION_REPEAT_CAP = 2

_THIRD_PARAM_PRIORITY = {
    "momentum": ["REBAL_WEEKS", "TOP_PCT", "MA_WEEKS", "MIN_HOLD_DAYS"],
    "superstock": ["max_positions", "vix_hard_cap", "vix_ma_multiplier", "breakout_extension_max"],
}

VALIDATOR_MODE_PROMOTED_REGION = "promoted_region_followup"
REGION_STATE_NEWLY_PROMOTED = "newly_promoted"
REGION_STATE_UNDER_FOLLOW_UP = "under_follow_up"
REGION_STATE_VALIDATED = "validated"
REGION_STATE_REJECTED = "rejected"
REGION_STATE_RETIRED = "retired"
ACTIVE_REGION_STATES = {
    REGION_STATE_NEWLY_PROMOTED,
    REGION_STATE_UNDER_FOLLOW_UP,
    REGION_STATE_VALIDATED,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _workspace_path(workspace_root: str | Path, relative: Path) -> Path:
    root = Path(workspace_root)
    return relative if relative.is_absolute() else root / relative


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return value
    return value


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if value != value:
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _default_region_evidence(reason: str = "stable grid-search plateau promoted") -> dict[str, Any]:
    return {
        "followup_proposal_count": 0,
        "followup_execution_count": 0,
        "viable_count": 0,
        "baseline_beating_count": 0,
        "neighborhood_execution_count": 0,
        "neighborhood_viable_count": 0,
        "neighborhood_baseline_beating_count": 0,
        "local_stability_confirmed": False,
        "rejection_reasons": [],
        "source_proposal_ids": [],
        "source_experiment_ids": [],
        "last_activity_timestamp": None,
        "last_state_change_reason": reason,
    }


def _default_budget_feedback(state: str = REGION_STATE_NEWLY_PROMOTED) -> dict[str, Any]:
    if state == REGION_STATE_VALIDATED:
        return {
            "earned_budget": 1,
            "budget_reason": "validated promoted region earned bounded follow-up budget",
            "lost_budget_reason": None,
        }
    if state in {REGION_STATE_REJECTED, REGION_STATE_RETIRED}:
        return {
            "earned_budget": 0,
            "budget_reason": None,
            "lost_budget_reason": f"{state} regions are inactive",
        }
    return {"earned_budget": 0, "budget_reason": None, "lost_budget_reason": None}


def _default_region_health() -> dict[str, Any]:
    return {
        "region_health_state": "fragile",
        "region_drift_score": 0.0,
        "neighborhood_outcome_summary": {
            "original_plateau_cells": 0,
            "followup_cells": 0,
            "neighborhood_cells": 0,
            "original_plateau_score": None,
            "followup_score_mean": None,
            "neighborhood_score_mean": None,
            "followup_viable_rate": 0.0,
            "neighborhood_viable_rate": 0.0,
            "followup_baseline_rate": 0.0,
            "neighborhood_baseline_rate": 0.0,
        },
        "budget_change_reason": "region has no follow-up health evidence yet",
    }


def _ensure_region_lifecycle(region: dict[str, Any]) -> dict[str, Any]:
    updated = dict(region)
    state = str(updated.get("state") or REGION_STATE_NEWLY_PROMOTED)
    updated["state"] = state
    evidence = _default_region_evidence()
    evidence.update(updated.get("evidence") or {})
    updated["evidence"] = evidence
    feedback = _default_budget_feedback(state)
    feedback.update(updated.get("budget_feedback") or {})
    updated["budget_feedback"] = feedback
    health = _default_region_health()
    health.update(updated.get("region_health") or {})
    if state in ACTIVE_REGION_STATES:
        evidence = updated.get("evidence") or {}
        no_followup_evidence = (
            int(evidence.get("followup_execution_count") or 0) == 0
            and int(evidence.get("neighborhood_execution_count") or 0) == 0
        )
        if no_followup_evidence and str(health.get("region_health_state") or "") == "fragile":
            health["region_health_state"] = "stable"
            health["budget_change_reason"] = "region has no follow-up health evidence yet"
    updated["region_health"] = health
    if state in {REGION_STATE_REJECTED, REGION_STATE_RETIRED}:
        updated["ready_for_follow_up"] = False
    return updated


def _region_id(payload: dict[str, Any]) -> str:
    token = json.dumps(
        {
            "family": payload.get("family"),
            "param_x": payload.get("param_x"),
            "param_y": payload.get("param_y"),
            "source_grid_search_id": payload.get("source_grid_search_id"),
            "recommended_values": payload.get("recommended_values"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]
    return f"region_{payload.get('family')}_{digest}"


def _cell_config(cell: dict[str, Any], base_config: dict[str, Any], param_x: str, param_y: str) -> dict[str, Any]:
    raw = cell.get("config_json")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    config = dict(base_config)
    config[param_x] = cell.get("param_x_value")
    config[param_y] = cell.get("param_y_value")
    return config


def promoted_region_from_grid_summary(
    summary: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    template_id: str | None = None,
) -> dict[str, Any] | None:
    stability = summary.get("stability") if isinstance(summary.get("stability"), dict) else {}
    if stability.get("region_class") != "stable_plateau":
        return None
    if stability.get("promotion_recommendation") != "promote_region":
        return None

    family = str(summary.get("family") or "").strip().lower()
    param_x = str(summary.get("param_x") or "").strip()
    param_y = str(summary.get("param_y") or "").strip()
    top_cells = [cell for cell in summary.get("top_cells", []) if isinstance(cell, dict)]
    if not family or not param_x or not param_y or not top_cells:
        return None

    threshold = float(stability.get("near_best_threshold") or float("-inf"))
    plateau_cells = [
        cell for cell in top_cells
        if float(cell.get("objective_score") or float("-inf")) >= threshold
    ] or top_cells[:1]
    x_values = [cell.get("param_x_value") for cell in plateau_cells]
    y_values = [cell.get("param_y_value") for cell in plateau_cells]
    best_cell = summary.get("best_cell") if isinstance(summary.get("best_cell"), dict) else top_cells[0]
    source_grid_search_id = Path(source_path).stem if source_path else str(summary.get("grid_search_id") or "")
    if source_grid_search_id.endswith(".summary"):
        source_grid_search_id = source_grid_search_id[:-8]

    artifact = {
        "region_id": None,
        "family": family,
        "template_id": template_id or summary.get("template_id") or f"{family}_grid_region",
        "param_x": param_x,
        "param_y": param_y,
        "promoted_bounds": {
            param_x: {"min": min(x_values), "max": max(x_values), "values": list(dict.fromkeys(x_values))},
            param_y: {"min": min(y_values), "max": max(y_values), "values": list(dict.fromkeys(y_values))},
        },
        "recommended_values": {
            param_x: best_cell.get("param_x_value"),
            param_y: best_cell.get("param_y_value"),
        },
        "top_cells": plateau_cells[:5],
        "plateau_metrics": {
            "plateau_width": stability.get("plateau_width"),
            "plateau_score": stability.get("plateau_score"),
            "neighbor_score_median": stability.get("neighbor_score_median"),
            "neighbor_sharpe_mean": stability.get("neighbor_sharpe_mean"),
            "neighbor_viable_fraction": stability.get("neighbor_viable_fraction"),
            "near_best_neighbor_fraction": stability.get("near_best_neighbor_fraction"),
        },
        "region_class": stability.get("region_class"),
        "promotion_recommendation": stability.get("promotion_recommendation"),
        "source_grid_search_id": source_grid_search_id,
        "source_summary_path": str(source_path) if source_path else summary.get("summary_path"),
        "ready_for_follow_up": True,
        "state": REGION_STATE_NEWLY_PROMOTED,
        "evidence": _default_region_evidence(),
        "budget_feedback": _default_budget_feedback(),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "base_config": summary.get("base_config") or {},
        "followup_controls": {
            "region_followup_budget": DEFAULT_REGION_FOLLOWUP_BUDGET,
            "max_region_followup_candidates": DEFAULT_MAX_REGION_FOLLOWUP_CANDIDATES,
            "region_followup_repeat_cap": DEFAULT_REGION_FOLLOWUP_REPEAT_CAP,
            "validated_region_neighbor_budget": DEFAULT_VALIDATED_REGION_NEIGHBOR_BUDGET,
            "validated_region_neighbor_radius": DEFAULT_VALIDATED_REGION_NEIGHBOR_RADIUS,
            "max_validated_region_neighbors": DEFAULT_MAX_VALIDATED_REGION_NEIGHBORS,
            "validated_region_repeat_cap": DEFAULT_VALIDATED_REGION_REPEAT_CAP,
            "third_parameter_expansion_enabled": True,
        },
    }
    artifact["region_id"] = _region_id(artifact)
    return _json_safe(artifact)


def save_promoted_region(region: dict[str, Any], *, workspace_root: str | Path = ".") -> Path:
    output_dir = _workspace_path(workspace_root, PROMOTED_REGIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    region = _ensure_region_lifecycle(region)
    region["updated_at"] = _utc_now()
    path = output_dir / f"{region['region_id']}.json"
    path.write_text(json.dumps(region, indent=2, sort_keys=True, default=str))
    index_path = _workspace_path(workspace_root, PROMOTED_REGIONS_INDEX)
    existing = load_promoted_regions(workspace_root=workspace_root, include_retired=True)
    by_id = {item["region_id"]: item for item in existing if item.get("region_id")}
    by_id[region["region_id"]] = region
    rows = sorted(by_id.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    index_path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    return path


def save_promoted_region_from_summary(
    summary: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    workspace_root: str | Path = ".",
    template_id: str | None = None,
) -> Path | None:
    region = promoted_region_from_grid_summary(summary, source_path=source_path, template_id=template_id)
    if region is None:
        return None
    return save_promoted_region(region, workspace_root=workspace_root)


def load_promoted_regions(
    *,
    workspace_root: str | Path = ".",
    include_retired: bool = False,
) -> list[dict[str, Any]]:
    index_path = _workspace_path(workspace_root, PROMOTED_REGIONS_INDEX)
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text())
    except json.JSONDecodeError:
        return []
    rows = payload if isinstance(payload, list) else []
    if include_retired:
        return [_ensure_region_lifecycle(row) for row in rows if isinstance(row, dict)]
    return [
        _ensure_region_lifecycle(row) for row in rows
        if isinstance(row, dict) and str(row.get("state") or REGION_STATE_NEWLY_PROMOTED) in ACTIVE_REGION_STATES
    ]


def active_promoted_regions(
    *,
    workspace_root: str | Path = ".",
    families: list[str] | None = None,
    limit: int = DEFAULT_MAX_ACTIVE_PROMOTED_REGIONS,
) -> list[dict[str, Any]]:
    family_set = {family.strip().lower() for family in families or [] if family}
    regions = [
        region for region in load_promoted_regions(workspace_root=workspace_root)
        if region.get("ready_for_follow_up")
        and region.get("region_class") == "stable_plateau"
        and region.get("promotion_recommendation") == "promote_region"
        and region.get("state") in ACTIVE_REGION_STATES
        and str((region.get("region_health") or {}).get("region_health_state") or "") in {"strengthening", "stable"}
        and (not family_set or region.get("family") in family_set)
    ]
    regions.sort(
        key=lambda item: (
            0 if str((item.get("region_health") or {}).get("region_health_state") or "") == "strengthening" else 1,
            0 if item.get("state") == REGION_STATE_VALIDATED else 1,
            -float((item.get("region_health") or {}).get("region_drift_score") or 0.0),
            -float((item.get("plateau_metrics") or {}).get("plateau_score") or 0.0),
            str(item.get("region_id") or ""),
        )
    )
    return regions[: max(0, int(limit))]


def _third_param_values(family: str, config: dict[str, Any], excluded: set[str], limit: int = 2) -> tuple[str | None, list[Any]]:
    space = get_family_search_space(family)
    for param in _THIRD_PARAM_PRIORITY.get(family, []):
        if param in excluded or param not in space:
            continue
        choices = list(space[param].get("choices") or [])
        if not choices:
            continue
        current = config.get(param, space[param].get("default"))
        values = [value for value in choices if value != current]
        if values:
            return param, values[:limit]
    return None, []


def _coerce_override_value(spec: dict[str, Any], value: Any) -> Any:
    if value is None:
        if spec.get("nullable", False):
            return None
        raise ValueError(f"{spec['name']} does not allow null.")
    value_type = spec["type"]
    if value_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be an int.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        raise ValueError(f"{spec['name']} must be an int.")
    if value_type == "float":
        if isinstance(value, bool):
            raise ValueError(f"{spec['name']} must be a float.")
        return round(float(value), 6)
    if value_type == "bool":
        if isinstance(value, bool):
            return value
        raise ValueError(f"{spec['name']} must be a bool.")
    if value_type == "categorical":
        return value
    raise ValueError(f"Unsupported type '{value_type}' for {spec['name']}.")


def _value_allowed(value: Any, allowed_values: list[Any]) -> bool:
    for allowed in allowed_values:
        if value == allowed:
            return True
        try:
            if float(value) == float(allowed):
                return True
        except (TypeError, ValueError):
            pass
    return False


def normalize_promoted_region_followup_config(
    family: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if metadata.get("validator_mode") != VALIDATOR_MODE_PROMOTED_REGION:
        raise ValueError("promoted-region validator mode is required.")
    if metadata.get("source_type") != "region_followup":
        raise ValueError("promoted-region override is only allowed for region_followup candidates.")
    if metadata.get("region_class") != "stable_plateau":
        raise ValueError("promoted-region override requires stable_plateau.")
    if metadata.get("promotion_recommendation") != "promote_region":
        raise ValueError("promoted-region override requires promote_region recommendation.")
    if not metadata.get("ready_for_follow_up"):
        raise ValueError("promoted-region override requires ready_for_follow_up.")
    if not metadata.get("source_region_id"):
        raise ValueError("promoted-region override requires source_region_id.")

    allowed_params = list(metadata.get("allowed_override_params") or [])
    allowed_values = dict(metadata.get("allowed_override_values") or {})
    if not allowed_params or len(allowed_params) > 2:
        raise ValueError("promoted-region override requires one or two explicit params.")

    space = get_family_search_space(family)
    unknown = sorted(set(config) - set(space))
    if unknown:
        raise ValueError(f"Unknown config key(s) for {family}: {', '.join(unknown)}")
    for param in allowed_params:
        if param not in space:
            raise ValueError(f"Unknown override param for {family}: {param}")
        if not isinstance(allowed_values.get(param), list) or not allowed_values[param]:
            raise ValueError(f"Missing allowed override values for {param}.")

    fixed = {key: value for key, value in config.items() if key not in set(allowed_params)}
    normalized = normalize_experiment_config(family, fixed)
    off_grid_params: list[str] = []
    for param in allowed_params:
        value = _coerce_override_value(space[param], config.get(param, normalized.get(param)))
        if not _value_allowed(value, allowed_values[param]):
            raise ValueError(f"{param}={value} is not authorized by promoted region.")
        choices = space[param].get("choices")
        if choices is not None and value not in choices:
            off_grid_params.append(param)
        normalized[param] = value

    return dict(sorted(normalized.items())), {
        "validator_mode": VALIDATOR_MODE_PROMOTED_REGION,
        "allowed_override_params": allowed_params,
        "allowed_override_values": allowed_values,
        "off_grid_params": sorted(off_grid_params),
        "validation_override_reason": (
            "stable promoted-region artifact authorized exact local off-grid values"
            if off_grid_params
            else "stable promoted-region follow-up used official-grid values"
        ),
    }


def _append_candidate(
    *,
    configs: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    seen: set[str],
    family: str,
    config: dict[str, Any],
    meta: dict[str, Any],
) -> bool:
    try:
        normalized = normalize_experiment_config(family, config)
    except ValueError:
        try:
            normalized, override_info = normalize_promoted_region_followup_config(family, config, meta)
        except ValueError:
            return False
        meta.update(override_info)
    config_hash = compute_config_hash(family, normalized)
    if config_hash in seen:
        return False
    seen.add(config_hash)
    configs.append(normalized)
    metadata.append(meta)
    return True


def _evidence_from_rows(rows: list[dict[str, Any]], current: dict[str, Any]) -> dict[str, Any]:
    evidence = _default_region_evidence(str(current.get("last_state_change_reason") or ""))
    evidence.update(current)
    executed_rows = [row for row in rows if str(row.get("status") or "").strip()]
    proposal_ids = {
        str(row.get("source_proposal_id"))
        for row in rows
        if str(row.get("source_proposal_id") or "").strip()
    }
    experiment_ids = {
        str(row.get("experiment_id"))
        for row in executed_rows
        if str(row.get("experiment_id") or "").strip()
    }
    viable_count = sum(1 for row in executed_rows if _truthy(row.get("viable")))
    baseline_count = sum(
        1 for row in executed_rows
        if _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))
    )
    neighborhood_rows = [
        row for row in executed_rows
        if str(row.get("targeted_follow_up_type") or row.get("exploration_mode") or "")
        == "validated_region_neighborhood"
    ]
    neighborhood_viable = sum(1 for row in neighborhood_rows if _truthy(row.get("viable")))
    neighborhood_baseline = sum(
        1 for row in neighborhood_rows
        if _truthy(row.get("beats_baseline_objective")) or _truthy(row.get("beats_baseline_guardrails"))
    )
    existing_proposal_ids = {
        str(value) for value in evidence.get("source_proposal_ids") or [] if str(value).strip()
    }
    existing_experiment_ids = {
        str(value) for value in evidence.get("source_experiment_ids") or [] if str(value).strip()
    }
    all_proposal_ids = sorted(existing_proposal_ids | proposal_ids)
    all_experiment_ids = sorted(existing_experiment_ids | experiment_ids)
    evidence.update({
        "followup_proposal_count": max(
            int(evidence.get("followup_proposal_count") or 0),
            len(all_proposal_ids),
        ),
        "followup_execution_count": len(executed_rows),
        "viable_count": viable_count,
        "baseline_beating_count": baseline_count,
        "neighborhood_execution_count": len(neighborhood_rows),
        "neighborhood_viable_count": neighborhood_viable,
        "neighborhood_baseline_beating_count": neighborhood_baseline,
        "local_stability_confirmed": viable_count >= 2 or (viable_count >= 1 and baseline_count >= 1),
        "source_proposal_ids": all_proposal_ids,
        "source_experiment_ids": all_experiment_ids,
    })
    if executed_rows:
        evidence["last_activity_timestamp"] = _utc_now()
    return evidence


def _region_plateau_score(region: dict[str, Any]) -> float:
    plateau_metrics = region.get("plateau_metrics") if isinstance(region.get("plateau_metrics"), dict) else {}
    plateau_score = plateau_metrics.get("plateau_score")
    if plateau_score is not None:
        try:
            return float(plateau_score)
        except (TypeError, ValueError):
            pass
    top_cells = [cell for cell in region.get("top_cells", []) if isinstance(cell, dict)]
    scores = []
    for cell in top_cells:
        try:
            scores.append(float(cell.get("objective_score") or 0.0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def _row_lane(row: dict[str, Any]) -> str:
    lane = str(
        row.get("targeted_follow_up_type")
        or row.get("exploration_mode")
        or row.get("proposal_role")
        or ""
    ).strip().lower()
    if lane == "validated_region_neighborhood":
        return "neighborhood"
    if lane in {"region_followup", "region_cell_confirmation", "neighbor_validation", "third_parameter_expansion"}:
        return "followup"
    return "followup"


def _build_region_health(region: dict[str, Any], evidence: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    plateau_score = _region_plateau_score(region)
    followup_rows = [row for row in rows if _row_lane(row) == "followup"]
    neighborhood_rows = [row for row in rows if _row_lane(row) == "neighborhood"]

    def _mean_score(items: list[dict[str, Any]]) -> float | None:
        scores: list[float] = []
        for item in items:
            try:
                score = item.get("objective_score")
                if score is None:
                    continue
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
        if not scores:
            return None
        return sum(scores) / len(scores)

    def _rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for item in items if _truthy(item.get(key))) / len(items)

    followup_score_mean = _mean_score(followup_rows)
    neighborhood_score_mean = _mean_score(neighborhood_rows)
    followup_viable_rate = _rate(followup_rows, "viable")
    neighborhood_viable_rate = _rate(neighborhood_rows, "viable")
    followup_baseline_rate = _rate(followup_rows, "beats_baseline_objective")
    neighborhood_baseline_rate = _rate(neighborhood_rows, "beats_baseline_objective")
    total_rows = len(followup_rows) + len(neighborhood_rows)

    if total_rows > 0:
        weighted_deltas: list[float] = []
        if followup_score_mean is not None:
            weighted_deltas.append((followup_score_mean - plateau_score) * max(1, len(followup_rows)))
        if neighborhood_score_mean is not None:
            weighted_deltas.append((neighborhood_score_mean - plateau_score) * max(1, len(neighborhood_rows)))
        region_drift_score = sum(weighted_deltas) / max(1, sum(max(1, len(items)) for items in (followup_rows, neighborhood_rows) if items))
    else:
        region_drift_score = 0.0

    if region.get("state") in {REGION_STATE_REJECTED, REGION_STATE_RETIRED}:
        health_state = "fragile"
        budget_change_reason = "inactive region does not receive active follow-up budget"
    elif total_rows == 0:
        health_state = "stable"
        budget_change_reason = "region has no follow-up health evidence yet"
    elif (
        region_drift_score >= 0.05
        and max(followup_viable_rate, neighborhood_viable_rate) >= 0.5
        and max(followup_baseline_rate, neighborhood_baseline_rate) >= 0.25
    ):
        health_state = "strengthening"
        budget_change_reason = "follow-up and neighborhood evidence improved on the original plateau"
    elif abs(region_drift_score) <= 0.05 and max(followup_viable_rate, neighborhood_viable_rate) > 0.0:
        health_state = "stable"
        budget_change_reason = "follow-up and neighborhood evidence stayed close to the original plateau"
    elif region_drift_score <= -0.05 or (followup_viable_rate == 0.0 and neighborhood_viable_rate == 0.0):
        health_state = "degrading"
        if neighborhood_rows and not followup_rows:
            budget_change_reason = "validated neighborhood follow-up lost viability"
        elif neighborhood_rows:
            budget_change_reason = "validated neighborhood follow-up lost viability"
        else:
            budget_change_reason = "follow-up evidence lost viability"
    else:
        health_state = "fragile"
        budget_change_reason = "follow-up evidence is too thin to support more active budget"

    neighborhood_outcome_summary = {
        "original_plateau_cells": len([cell for cell in region.get("top_cells", []) if isinstance(cell, dict)]),
        "followup_cells": len(followup_rows),
        "neighborhood_cells": len(neighborhood_rows),
        "original_plateau_score": round(float(plateau_score), 6),
        "followup_score_mean": None if followup_score_mean is None else round(float(followup_score_mean), 6),
        "neighborhood_score_mean": None if neighborhood_score_mean is None else round(float(neighborhood_score_mean), 6),
        "followup_viable_rate": round(float(followup_viable_rate), 6),
        "neighborhood_viable_rate": round(float(neighborhood_viable_rate), 6),
        "followup_baseline_rate": round(float(followup_baseline_rate), 6),
        "neighborhood_baseline_rate": round(float(neighborhood_baseline_rate), 6),
    }

    return {
        "region_health_state": health_state,
        "region_drift_score": round(float(region_drift_score), 6),
        "neighborhood_outcome_summary": neighborhood_outcome_summary,
        "budget_change_reason": budget_change_reason,
    }


def _transition_region(region: dict[str, Any], evidence: dict[str, Any], health: dict[str, Any]) -> tuple[str, str]:
    current = str(region.get("state") or REGION_STATE_NEWLY_PROMOTED)
    if current == REGION_STATE_RETIRED:
        return current, "retired regions remain inactive"
    if current == REGION_STATE_REJECTED:
        return current, "rejected regions remain inactive"

    executions = int(evidence.get("followup_execution_count") or 0)
    proposals = int(evidence.get("followup_proposal_count") or 0)
    viable = int(evidence.get("viable_count") or 0)
    baseline = int(evidence.get("baseline_beating_count") or 0)
    stable = bool(evidence.get("local_stability_confirmed"))
    neighborhood_executions = int(evidence.get("neighborhood_execution_count") or 0)
    neighborhood_viable = int(evidence.get("neighborhood_viable_count") or 0)
    health_state = str(health.get("region_health_state") or "fragile")

    if health_state in {"fragile", "degrading"} and executions >= 2 and viable == 0 and neighborhood_executions == 0:
        return REGION_STATE_REJECTED, "fragile region lost bounded follow-up viability"
    if health_state == "degrading" and current == REGION_STATE_VALIDATED:
        return REGION_STATE_UNDER_FOLLOW_UP, "validated region degraded and fell back to follow-up"
    if health_state in {"strengthening", "stable"} and (stable or viable >= 1 or baseline >= 1):
        return REGION_STATE_VALIDATED, "follow-up and neighborhood evidence confirmed bounded stability"
    if current == REGION_STATE_VALIDATED and neighborhood_executions >= 2 and neighborhood_viable == 0:
        return REGION_STATE_UNDER_FOLLOW_UP, "validated neighborhood follow-up degraded with no viable cells"
    if proposals > 0 or executions > 0:
        return REGION_STATE_UNDER_FOLLOW_UP, "bounded region follow-up is in progress"
    return REGION_STATE_NEWLY_PROMOTED, "awaiting bounded region follow-up"


def _budget_feedback_for_state(state: str, evidence: dict[str, Any], health: dict[str, Any]) -> dict[str, Any]:
    health_state = str(health.get("region_health_state") or "fragile")
    drift_score = float(health.get("region_drift_score") or 0.0)
    if state in {REGION_STATE_REJECTED, REGION_STATE_RETIRED} or health_state == "fragile":
        return {
            "earned_budget": 0,
            "budget_reason": None,
            "lost_budget_reason": health.get("budget_change_reason") or f"{state} regions do not receive active follow-up budget",
            "budget_change_reason": health.get("budget_change_reason"),
        }
    if health_state == "strengthening":
        viable = int(evidence.get("viable_count") or 0)
        baseline = int(evidence.get("baseline_beating_count") or 0)
        earned = min(2, max(1, baseline, viable // 2))
        return {
            "earned_budget": earned,
            "budget_reason": "validated promoted region strengthened enough to retain or slightly improve bounded budget",
            "lost_budget_reason": None,
            "budget_change_reason": f"strengthening drift={drift_score:.3f}",
        }
    if health_state == "stable":
        return {
            "earned_budget": 1,
            "budget_reason": "validated promoted region stayed stable and keeps a small bounded budget slice",
            "lost_budget_reason": None,
            "budget_change_reason": f"stable drift={drift_score:.3f}",
        }
    if health_state == "degrading":
        return {
            "earned_budget": 0,
            "budget_reason": None,
            "lost_budget_reason": health.get("budget_change_reason") or "degrading region lost earned budget and fell back to the base lane",
            "budget_change_reason": health.get("budget_change_reason") or f"degrading drift={drift_score:.3f}",
        }
    return {
        "earned_budget": 0,
        "budget_reason": "region remains bounded to the base follow-up lane",
        "lost_budget_reason": None,
        "budget_change_reason": health.get("budget_change_reason"),
    }


def update_promoted_region_from_followup_rows(
    region: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = _ensure_region_lifecycle(region)
    evidence = _evidence_from_rows(rows, updated.get("evidence") or {})
    health = _build_region_health(updated, evidence, rows)
    new_state, reason = _transition_region(updated, evidence, health)
    rejection_reasons = list(evidence.get("rejection_reasons") or [])
    if new_state == REGION_STATE_REJECTED and reason not in rejection_reasons:
        rejection_reasons.append(reason)
    evidence["rejection_reasons"] = rejection_reasons
    if new_state != updated.get("state") or reason != evidence.get("last_state_change_reason"):
        evidence["last_state_change_reason"] = reason
        evidence["last_activity_timestamp"] = _utc_now()
    updated["state"] = new_state
    updated["evidence"] = evidence
    updated["region_health"] = health
    updated["budget_feedback"] = _budget_feedback_for_state(new_state, evidence, health)
    updated["ready_for_follow_up"] = (
        bool(updated.get("ready_for_follow_up"))
        and new_state in ACTIVE_REGION_STATES
        and str(health.get("region_health_state") or "") in {"strengthening", "stable"}
    )
    updated["updated_at"] = _utc_now()
    return _json_safe(updated)


def refresh_promoted_region_states(
    *,
    workspace_root: str | Path = ".",
    base_dir: str | Path = "experiments",
) -> list[dict[str, Any]]:
    regions = load_promoted_regions(workspace_root=workspace_root, include_retired=True)
    if not regions:
        return []
    index_path = _workspace_path(workspace_root, Path(base_dir))
    index = load_results_index(str(index_path))
    rows_by_region: dict[str, list[dict[str, Any]]] = {}
    if "source_region_id" in index.columns:
        region_rows = index[index["source_region_id"].notna()]
        for row in region_rows.to_dict("records"):
            region_id = str(row.get("source_region_id") or "")
            if region_id:
                rows_by_region.setdefault(region_id, []).append(row)

    refreshed: list[dict[str, Any]] = []
    for region in regions:
        updated = update_promoted_region_from_followup_rows(
            region,
            rows_by_region.get(str(region.get("region_id") or ""), []),
        )
        save_promoted_region(updated, workspace_root=workspace_root)
        refreshed.append(updated)
    build_promoted_regions_report(workspace_root=workspace_root)
    return refreshed


def record_region_followup_proposal(
    *,
    workspace_root: str | Path = ".",
    proposal_id: str,
    lane_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    active_ids = {
        str(region.get("region_id"))
        for region in lane_summary.get("regions", [])
        if int(region.get("candidate_count") or 0) > 0 and str(region.get("region_id") or "")
    }
    if not active_ids:
        return []
    updated_regions: list[dict[str, Any]] = []
    for region in load_promoted_regions(workspace_root=workspace_root, include_retired=True):
        if str(region.get("region_id") or "") not in active_ids:
            continue
        updated = _ensure_region_lifecycle(region)
        evidence = dict(updated.get("evidence") or {})
        proposal_ids = list(evidence.get("source_proposal_ids") or [])
        if proposal_id not in proposal_ids:
            proposal_ids.append(proposal_id)
            evidence["followup_proposal_count"] = int(evidence.get("followup_proposal_count") or 0) + 1
        evidence["source_proposal_ids"] = proposal_ids
        evidence["last_activity_timestamp"] = _utc_now()
        if updated.get("state") == REGION_STATE_NEWLY_PROMOTED:
            updated["state"] = REGION_STATE_UNDER_FOLLOW_UP
            evidence["last_state_change_reason"] = "bounded region follow-up proposal generated"
        updated["evidence"] = evidence
        health = _build_region_health(updated, evidence, [])
        updated["region_health"] = health
        updated["budget_feedback"] = _budget_feedback_for_state(str(updated.get("state")), evidence, health)
        save_promoted_region(updated, workspace_root=workspace_root)
        updated_regions.append(updated)
    build_promoted_regions_report(workspace_root=workspace_root)
    return updated_regions


def compute_region_budget_feedback(
    regions: list[dict[str, Any]],
    *,
    base_budget: int,
    max_boost: int = 2,
) -> dict[str, Any]:
    base = max(0, int(base_budget))
    max_boost = max(0, int(max_boost))
    earners = []
    losses = []
    for region in regions:
        budget_feedback = region.get("budget_feedback") or {}
        health = region.get("region_health") or {}
        health_state = str(health.get("region_health_state") or "").strip().lower()
        earned_budget = int(budget_feedback.get("earned_budget") or 0)
        lost_budget_reason = budget_feedback.get("lost_budget_reason")
        if region.get("state") == REGION_STATE_VALIDATED and health_state in {"strengthening", "stable"} and earned_budget > 0:
            earners.append({
                "region_id": region.get("region_id"),
                "state": region.get("state"),
                "region_health_state": health_state,
                "earned_budget": earned_budget,
                "reason": budget_feedback.get("budget_reason"),
            })
        if region.get("state") in {REGION_STATE_REJECTED, REGION_STATE_RETIRED} or health_state == "fragile":
            losses.append({
                "region_id": region.get("region_id"),
                "state": region.get("state"),
                "region_health_state": health_state or "fragile",
                "lost_budget_reason": lost_budget_reason or "inactive region",
            })
        elif health_state == "degrading" and lost_budget_reason:
            losses.append({
                "region_id": region.get("region_id"),
                "state": region.get("state"),
                "region_health_state": health_state,
                "lost_budget_reason": lost_budget_reason,
            })
    boost = min(max_boost, sum(item["earned_budget"] for item in earners))
    loss = min(base, len(losses))
    if boost and not loss:
        budget_change_reason = "validated strengthening/stable regions earned a bounded budget boost"
    elif boost and loss:
        budget_change_reason = "validated regions earned budget, but fragile/degrading regions also reduced active follow-up"
    elif loss:
        budget_change_reason = "fragile/degrading regions reduced active follow-up budget"
    else:
        budget_change_reason = "no validated promoted-region budget boost applied"
    return {
        "base_budget": base,
        "earned_budget_boost": boost,
        "lost_budget": loss,
        "effective_budget": max(0, base + boost - loss),
        "max_boost": max_boost,
        "earning_regions": earners,
        "losing_regions": losses,
        "budget_change_reason": budget_change_reason,
        "rationale": budget_change_reason,
    }


def _sort_values(values: list[Any]) -> list[Any]:
    try:
        return sorted(values, key=lambda value: (float(value), str(value)))
    except (TypeError, ValueError):
        return sorted(values, key=str)


def _neighbor_values(values: list[Any], current: Any, radius: int) -> list[Any]:
    ordered = _sort_values(list(dict.fromkeys(values)))
    if current not in ordered:
        return []
    index = ordered.index(current)
    lower = max(0, index - max(0, int(radius)))
    upper = min(len(ordered), index + max(0, int(radius)) + 1)
    return [value for value in ordered[lower:upper] if value != current]


def _config_hashes_by_family(configs_by_family: dict[str, list[dict[str, Any]]] | None) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = {}
    for family, configs in (configs_by_family or {}).items():
        family_key = str(family or "").strip().lower()
        if not family_key:
            continue
        hashes[family_key] = {
            compute_config_hash(family_key, config)
            for config in configs
            if isinstance(config, dict)
        }
    return hashes


def generate_validated_region_neighborhood_candidates(
    regions: list[dict[str, Any]],
    *,
    budget: int = DEFAULT_VALIDATED_REGION_NEIGHBOR_BUDGET,
    neighbor_radius: int = DEFAULT_VALIDATED_REGION_NEIGHBOR_RADIUS,
    max_neighbors: int = DEFAULT_MAX_VALIDATED_REGION_NEIGHBORS,
    repeat_cap: int = DEFAULT_VALIDATED_REGION_REPEAT_CAP,
    enable_third_parameter: bool = True,
    existing_configs_by_family: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    budget = max(0, int(budget))
    max_neighbors = max(0, int(max_neighbors))
    repeat_cap = max(1, int(repeat_cap))
    configs_by_family: dict[str, list[dict[str, Any]]] = {}
    metadata_by_family: dict[str, list[dict[str, Any]]] = {}
    seen_by_family = _config_hashes_by_family(existing_configs_by_family)
    lane_regions: list[dict[str, Any]] = []

    for region in regions:
        if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_neighbors:
            break
        if region.get("state") != REGION_STATE_VALIDATED:
            continue
        family = str(region.get("family") or "").strip().lower()
        if not family:
            continue
        param_x = str(region.get("param_x") or "")
        param_y = str(region.get("param_y") or "")
        promoted_bounds = region.get("promoted_bounds") if isinstance(region.get("promoted_bounds"), dict) else {}
        x_values = list((promoted_bounds.get(param_x) or {}).get("values") or [])
        y_values = list((promoted_bounds.get(param_y) or {}).get("values") or [])
        if not x_values or not y_values:
            continue

        base_config = dict(region.get("base_config") or {})
        cells = [cell for cell in region.get("top_cells", []) if isinstance(cell, dict)]
        family_configs = configs_by_family.setdefault(family, [])
        family_metadata = metadata_by_family.setdefault(family, [])
        seen = seen_by_family.setdefault(family, set())
        region_start_count = len(family_configs)

        for cell in cells[:repeat_cap]:
            if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_neighbors:
                break
            cell_config = _cell_config(cell, base_config, param_x, param_y)
            for param, values in ((param_x, x_values), (param_y, y_values)):
                current = cell_config.get(param)
                for value in _neighbor_values(values, current, neighbor_radius):
                    if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_neighbors:
                        break
                    before = len(family_configs)
                    config = dict(cell_config)
                    config[param] = value
                    meta = _candidate_metadata(region, "validated_region_neighbor", [cell])
                    meta.update({
                        "proposal_role": "validated_region_neighbor",
                        "exploration_mode": "validated_region_neighborhood",
                        "targeted_follow_up_type": "validated_region_neighborhood",
                        "neighbor_of_cell": cell,
                        "neighbor_generation_reason": (
                            f"validated region neighborhood: {param} {current} -> {value}"
                        ),
                        "neighbor_grid_status": _neighbor_grid_status(family, config, [param_x, param_y]),
                    })
                    _append_candidate(
                        configs=family_configs,
                        metadata=family_metadata,
                        seen=seen,
                        family=family,
                        config=config,
                        meta=meta,
                    )
                    if len(family_configs) > before:
                        budget -= 1

        if enable_third_parameter and budget > 0 and sum(len(v) for v in configs_by_family.values()) < max_neighbors and cells:
            best_config = _cell_config(cells[0], base_config, param_x, param_y)
            third_param, third_values = _third_param_values(family, best_config, {param_x, param_y}, limit=1)
            for value in third_values:
                if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_neighbors:
                    break
                before = len(family_configs)
                config = dict(best_config)
                config[third_param] = value
                meta = _candidate_metadata(region, "validated_region_third_parameter", [cells[0]])
                meta.update({
                    "proposal_role": "validated_region_third_parameter",
                    "exploration_mode": "validated_region_neighborhood",
                    "targeted_follow_up_type": "validated_region_neighborhood",
                    "expanded_param": third_param,
                    "expanded_value": value,
                    "neighbor_of_cell": cells[0],
                    "neighbor_generation_reason": (
                        f"validated region bounded third-parameter follow-up: {third_param}={value}"
                    ),
                    "neighbor_grid_status": _neighbor_grid_status(family, config, [param_x, param_y, third_param]),
                })
                _append_candidate(
                    configs=family_configs,
                    metadata=family_metadata,
                    seen=seen,
                    family=family,
                    config=config,
                    meta=meta,
                )
                if len(family_configs) > before:
                    budget -= 1

        lane_regions.append({
            "region_id": region.get("region_id"),
            "family": family,
            "state": region.get("state"),
            "candidate_count": len(family_configs) - region_start_count,
            "evidence": region.get("evidence") or {},
            "budget_feedback": region.get("budget_feedback") or {},
        })

    summary = {
        "active": bool(any(configs_by_family.values())),
        "lane": "validated_region_neighborhood",
        "regions_considered": len([region for region in regions if region.get("state") == REGION_STATE_VALIDATED]),
        "candidate_count": sum(len(v) for v in configs_by_family.values()),
        "budget_remaining": budget,
        "neighbor_radius": neighbor_radius,
        "max_neighbors": max_neighbors,
        "repeat_cap": repeat_cap,
        "third_parameter_expansion_enabled": bool(enable_third_parameter),
        "regions": lane_regions,
        "rationale": (
            "validated promoted regions generated bounded neighborhood candidates"
            if any(configs_by_family.values())
            else "no validated promoted regions available for neighborhood search"
        ),
    }
    return configs_by_family, metadata_by_family, summary


def _neighbor_grid_status(family: str, config: dict[str, Any], params: list[str]) -> dict[str, str]:
    space = get_family_search_space(family)
    status: dict[str, str] = {}
    for param in params:
        choices = (space.get(param) or {}).get("choices")
        if choices is None:
            status[param] = "range_or_default"
        elif config.get(param) in choices:
            status[param] = "official_grid"
        else:
            status[param] = "promoted_off_grid"
    return status


def generate_region_followup_candidates(
    regions: list[dict[str, Any]],
    *,
    budget: int = DEFAULT_REGION_FOLLOWUP_BUDGET,
    max_candidates: int = DEFAULT_MAX_REGION_FOLLOWUP_CANDIDATES,
    repeat_cap: int = DEFAULT_REGION_FOLLOWUP_REPEAT_CAP,
    enable_third_parameter: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    budget = max(0, int(budget))
    max_candidates = max(0, int(max_candidates))
    repeat_cap = max(1, int(repeat_cap))
    configs_by_family: dict[str, list[dict[str, Any]]] = {}
    metadata_by_family: dict[str, list[dict[str, Any]]] = {}
    seen_by_family: dict[str, set[str]] = {}
    lane_regions: list[dict[str, Any]] = []

    for region in regions:
        if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_candidates:
            break
        if region.get("state") not in ACTIVE_REGION_STATES:
            continue
        family = str(region.get("family") or "").strip().lower()
        if not family:
            continue
        param_x = str(region.get("param_x") or "")
        param_y = str(region.get("param_y") or "")
        base_config = dict(region.get("base_config") or {})
        cells = [cell for cell in region.get("top_cells", []) if isinstance(cell, dict)]
        if not cells:
            continue
        family_configs = configs_by_family.setdefault(family, [])
        family_metadata = metadata_by_family.setdefault(family, [])
        seen = seen_by_family.setdefault(family, set())
        region_start_count = len(family_configs)

        for cell in cells[:repeat_cap]:
            if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_candidates:
                break
            before = len(family_configs)
            config = _cell_config(cell, base_config, param_x, param_y)
            _append_candidate(
                configs=family_configs,
                metadata=family_metadata,
                seen=seen,
                family=family,
                config=config,
                meta=_candidate_metadata(region, "region_cell_confirmation", [cell]),
            )
            if len(family_configs) > before:
                budget -= 1

        for cell in cells[repeat_cap: max(repeat_cap + 2, 3)]:
            if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_candidates:
                break
            before = len(family_configs)
            config = _cell_config(cell, base_config, param_x, param_y)
            _append_candidate(
                configs=family_configs,
                metadata=family_metadata,
                seen=seen,
                family=family,
                config=config,
                meta=_candidate_metadata(region, "neighbor_validation", [cell]),
            )
            if len(family_configs) > before:
                budget -= 1

        if enable_third_parameter and budget > 0 and sum(len(v) for v in configs_by_family.values()) < max_candidates:
            best_config = _cell_config(cells[0], base_config, param_x, param_y)
            third_param, third_values = _third_param_values(family, best_config, {param_x, param_y})
            for value in third_values:
                if budget <= 0 or sum(len(v) for v in configs_by_family.values()) >= max_candidates:
                    break
                before = len(family_configs)
                config = dict(best_config)
                config[third_param] = value
                meta = _candidate_metadata(region, "third_parameter_expansion", [cells[0]])
                meta["expanded_param"] = third_param
                meta["expanded_value"] = value
                _append_candidate(
                    configs=family_configs,
                    metadata=family_metadata,
                    seen=seen,
                    family=family,
                    config=config,
                    meta=meta,
                )
                if len(family_configs) > before:
                    budget -= 1

        lane_regions.append({
            "region_id": region.get("region_id"),
            "family": family,
            "state": region.get("state"),
            "region_class": region.get("region_class"),
            "promotion_recommendation": region.get("promotion_recommendation"),
            "evidence": region.get("evidence") or {},
            "budget_feedback": region.get("budget_feedback") or {},
            "candidate_count": len(family_configs) - region_start_count,
        })

    summary = {
        "active": bool(any(configs_by_family.values())),
        "lane": "region_followup",
        "regions_considered": len(regions),
        "candidate_count": sum(len(v) for v in configs_by_family.values()),
        "budget_remaining": budget,
        "max_candidates": max_candidates,
        "repeat_cap": repeat_cap,
        "third_parameter_expansion_enabled": bool(enable_third_parameter),
        "regions": lane_regions,
        "rationale": (
            "stable promoted grid-search regions generated bounded deterministic follow-up candidates"
            if any(configs_by_family.values())
            else "no stable promoted regions available for follow-up"
        ),
    }
    return configs_by_family, metadata_by_family, summary


def _candidate_metadata(region: dict[str, Any], kind: str, source_cells: list[dict[str, Any]]) -> dict[str, Any]:
    region_id = region.get("region_id")
    promoted_bounds = region.get("promoted_bounds") if isinstance(region.get("promoted_bounds"), dict) else {}
    allowed_params = [param for param in (region.get("param_x"), region.get("param_y")) if param]
    allowed_values = {
        param: list((promoted_bounds.get(param) or {}).get("values") or [])
        for param in allowed_params
    }
    return {
        "source_type": "region_followup",
        "strategy_type": "classical",
        "template_id": region.get("template_id"),
        "reason_selected": f"promoted grid-search region follow-up: {kind}",
        "exploration_mode": "region_followup",
        "proposal_role": kind,
        "region_label": region_id,
        "source_region_id": region_id,
        "source_grid_search_id": region.get("source_grid_search_id"),
        "source_cells": source_cells,
        "region_class": region.get("region_class"),
        "promotion_recommendation": region.get("promotion_recommendation"),
        "region_state": region.get("state"),
        "region_budget_feedback": region.get("budget_feedback") or {},
        "validator_mode": VALIDATOR_MODE_PROMOTED_REGION,
        "allowed_override_params": allowed_params,
        "allowed_override_values": allowed_values,
        "ready_for_follow_up": bool(region.get("ready_for_follow_up")),
        "validation_override_reason": "stable promoted region may authorize exact local off-grid values",
        "targeted_follow_up_required": True,
        "targeted_follow_up_type": "region_followup",
        "targeted_follow_up_reason": "stable grid-search plateau promoted for bounded follow-up",
        "holdout_horizon_tags": ["region_followup"],
        "holdout_regime_tags": [],
        "is_branch_repeat": True,
        "repeat_branch_flag": True,
    }


def build_promoted_regions_report(*, workspace_root: str | Path = ".") -> dict[str, Any]:
    regions = load_promoted_regions(workspace_root=workspace_root, include_retired=True)
    by_family: dict[str, list[dict[str, Any]]] = {}
    by_state: dict[str, int] = {}
    by_health_state: dict[str, int] = {}
    active: list[dict[str, Any]] = []
    validated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for region in regions:
        family = str(region.get("family") or "unknown")
        by_family.setdefault(family, []).append(region)
        state = str(region.get("state") or "")
        by_state[state] = by_state.get(state, 0) + 1
        health_state = str((region.get("region_health") or {}).get("region_health_state") or "fragile")
        by_health_state[health_state] = by_health_state.get(health_state, 0) + 1
        if region.get("ready_for_follow_up") and state in ACTIVE_REGION_STATES:
            active.append(region)
        if state == "validated":
            validated.append(region)
        if state in {"rejected", "rejected_fragile", "retired"}:
            rejected.append(region)
    report = {
        "generated_at": _utc_now(),
        "promoted_regions_by_family": {family: len(items) for family, items in by_family.items()},
        "promoted_regions_by_state": by_state,
        "promoted_regions_by_health_state": by_health_state,
        "active_region_followup_workload": [
            {
                "region_id": region.get("region_id"),
                "family": region.get("family"),
                "state": region.get("state"),
                "region_health_state": (region.get("region_health") or {}).get("region_health_state"),
                "region_drift_score": (region.get("region_health") or {}).get("region_drift_score"),
                "plateau_score": (region.get("plateau_metrics") or {}).get("plateau_score"),
                "evidence": region.get("evidence") or {},
                "budget_feedback": region.get("budget_feedback") or {},
            }
            for region in active
        ],
        "best_validated_regions": sorted(
            validated,
            key=lambda item: (
                0 if str((item.get("region_health") or {}).get("region_health_state") or "") == "strengthening" else 1,
                -float((item.get("region_health") or {}).get("region_drift_score") or 0.0),
                -float((item.get("plateau_metrics") or {}).get("plateau_score") or 0.0),
            ),
        )[:5],
        "rejected_fragile_regions": [
            {
                "region_id": region.get("region_id"),
                "family": region.get("family"),
                "state": region.get("state"),
                "region_health_state": (region.get("region_health") or {}).get("region_health_state"),
                "budget_feedback": region.get("budget_feedback") or {},
            }
            for region in rejected
        ],
        "total_regions": len(regions),
    }
    output_path = _workspace_path(workspace_root, PROMOTED_REGIONS_REPORT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
