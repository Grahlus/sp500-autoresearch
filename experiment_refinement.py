from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from agents.schemas import load_idea_records, load_latest_analysis_report
from experiment_batch import DEFAULT_BASELINES
from experiment_idea_library import expand_template_candidates, load_external_idea_seeds
from experiment_memory import load_research_memory, save_research_memory, update_family_memory
from experiment_novelty import coarse_signature_key, score_candidate, signature_distance
from experiment_spaces import (
    get_family_search_space,
    list_searchable_families,
    normalize_experiment_config,
    sample_random_candidates,
)
from experiment_store import compute_config_hash, load_prior_results, save_proposal_result
from experiment_types import ProposalRequest, ProposalResult


def build_proposal_request(
    *,
    strategy_families: list[str],
    source_batch_ids: list[str] | None = None,
    objective_name: str = "wf_v1_score",
    baseline_name: str | None = None,
    seed: int = 42,
    exploration_fraction: float = 0.65,
    exploitation_fraction: float = 0.35,
    max_experiments: int = 20,
    per_family_budgets: dict[str, int] | None = None,
    resume: bool = True,
    proposal_id: str | None = None,
    timestamp_utc: str | None = None,
    novelty_floor: float = 0.15,
    template_fraction: float = 0.50,
    cross_family_fraction: float = 0.20,
    max_near_duplicate_distance: int = 1,
    stagnation_escape_batches: int = 3,
    allow_external_seeds: bool = False,
    source_idea_ids: list[str] | None = None,
    use_idea_queue: bool = True,
    use_analysis_guidance: bool = True,
    min_viable_fill_rate: float = 0.50,
    min_viable_candidates: int | None = None,
    large_search_threshold: int = 50,
    min_large_search_candidates: int = 48,
    quality_gate: bool = True,
) -> ProposalRequest:
    families = [family.strip().lower() for family in strategy_families if family and family.strip()]
    unknown = sorted(set(families) - set(list_searchable_families()))
    if unknown:
        raise ValueError(f"Unknown strategy family(s): {', '.join(unknown)}")
    if max_experiments <= 0:
        raise ValueError("max_experiments must be > 0.")
    if exploration_fraction < 0 or exploitation_fraction < 0:
        raise ValueError("exploration_fraction and exploitation_fraction must be >= 0.")
    if round(exploration_fraction + exploitation_fraction, 6) != 1.0:
        raise ValueError("exploration_fraction + exploitation_fraction must equal 1.0.")
    if min_viable_fill_rate < 0 or min_viable_fill_rate > 1:
        raise ValueError("min_viable_fill_rate must be between 0 and 1.")
    if min_viable_candidates is not None and min_viable_candidates < 0:
        raise ValueError("min_viable_candidates must be >= 0.")
    if large_search_threshold < 1:
        raise ValueError("large_search_threshold must be > 0.")
    if min_large_search_candidates < 0:
        raise ValueError("min_large_search_candidates must be >= 0.")
    return ProposalRequest(
        proposal_id=proposal_id or datetime.now(UTC).strftime("proposal_%Y%m%d_%H%M%S"),
        timestamp_utc=timestamp_utc or datetime.now(UTC).isoformat(),
        source_batch_ids=source_batch_ids or [],
        strategy_families=families,
        objective_name=objective_name,
        baseline_name=baseline_name,
        seed=int(seed),
        exploration_fraction=float(exploration_fraction),
        exploitation_fraction=float(exploitation_fraction),
        max_experiments=int(max_experiments),
        per_family_budgets=per_family_budgets,
        resume=bool(resume),
        novelty_floor=float(novelty_floor),
        template_fraction=float(template_fraction),
        cross_family_fraction=float(cross_family_fraction),
        max_near_duplicate_distance=int(max_near_duplicate_distance),
        stagnation_escape_batches=int(stagnation_escape_batches),
        allow_external_seeds=bool(allow_external_seeds),
        source_idea_ids=list(source_idea_ids or []),
        use_idea_queue=bool(use_idea_queue),
        use_analysis_guidance=bool(use_analysis_guidance),
        min_viable_fill_rate=float(min_viable_fill_rate),
        min_viable_candidates=None if min_viable_candidates is None else int(min_viable_candidates),
        large_search_threshold=int(large_search_threshold),
        min_large_search_candidates=int(min_large_search_candidates),
        quality_gate=bool(quality_gate),
    )


def _helper_workspace_root(base_dir: str) -> str:
    path = Path(base_dir)
    if path.name == "experiments":
        return str(path.parent or Path("."))
    return str(path)


def _load_helper_ideas(
    *,
    request: ProposalRequest,
    base_dir: str,
) -> list[dict[str, Any]]:
    if not getattr(request, "use_idea_queue", True):
        return []
    workspace_root = _helper_workspace_root(base_dir)
    records = load_idea_records(workspace_root)
    filtered = [
        record
        for record in records
        if record.get("status", "new") == "new"
        and (record.get("family") in request.strategy_families or record.get("family") is None)
    ]
    filtered.sort(
        key=lambda record: (
            -float(record.get("priority") or 0.0),
            -float(record.get("novelty_score") or 0.0),
            str(record.get("idea_id") or ""),
        )
    )
    return filtered


def _latest_analysis_guidance(
    *,
    request: ProposalRequest,
    base_dir: str,
) -> dict[str, Any] | None:
    if not getattr(request, "use_analysis_guidance", True):
        return None
    workspace_root = _helper_workspace_root(base_dir)
    report = load_latest_analysis_report(workspace_root)
    if not report:
        return None
    return report


def _load_detailed_history(family: str, base_dir: str) -> pd.DataFrame:
    index = load_prior_results(family=family, base_dir=base_dir)
    rows: list[dict[str, Any]] = []
    if index.empty:
        return pd.DataFrame()

    for _, row in index.iterrows():
        result_dir = row.get("result_dir")
        if not isinstance(result_dir, str) or not result_dir:
            continue
        result_path = Path(result_dir) / "result.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text())
        spec = payload.get("spec", {})
        metrics = payload.get("metrics", {})
        robustness = payload.get("robustness", {})
        baseline = payload.get("baseline_comparison") or {}
        config = spec.get("config", spec.get("params", {}))
        entry = {
            "experiment_id": spec.get("experiment_id"),
            "batch_id": spec.get("batch_id"),
            "family": spec.get("strategy_family", spec.get("family")),
            "config_hash": spec.get("config_hash"),
            "config": config,
            "status": payload.get("status"),
            "objective_score": payload.get("objective_score"),
            "sharpe": metrics.get("sharpe"),
            "calmar": metrics.get("calmar"),
            "total_return": metrics.get("total_return"),
            "trades_per_year": metrics.get("trades_per_year"),
            "negative_windows": robustness.get("negative_windows"),
            "viable": robustness.get("viable"),
            "comparison_status": baseline.get("comparison_status"),
            "beats_baseline_objective": baseline.get("beats_baseline_objective"),
            "delta_sharpe": baseline.get("delta_sharpe"),
        }
        for key, value in config.items():
            entry[f"param__{key}"] = value
        rows.append(entry)

    return pd.DataFrame(rows)


def _value_summary(history: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if history.empty:
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    param_columns = [col for col in history.columns if col.startswith("param__")]
    for column in param_columns:
        param = column.removeprefix("param__")
        values: dict[str, Any] = {}
        for value, group in history.groupby(column, dropna=False):
            key = str(value)
            values[key] = {
                "count": int(len(group)),
                "avg_objective": None if group["objective_score"].isna().all() else float(group["objective_score"].fillna(0).mean()),
                "viable_rate": float(group["viable"].fillna(False).mean()) if "viable" in group else 0.0,
                "invalid_rate": float((group["status"] == "invalid").mean()),
                "no_trade_rate": float((group["status"] == "no_trades").mean()),
            }
        summaries[param] = values
    return summaries


def _under_tested_regions(value_summary: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    regions: dict[str, list[str]] = {}
    for param, values in value_summary.items():
        sparse = [value for value, stats in values.items() if int(stats.get("count", 0)) < 2]
        if sparse:
            regions[param] = sparse
    return regions


def _poor_region_signatures(history: pd.DataFrame, family: str, median_score: float) -> set[str]:
    if history.empty:
        return set()
    poor: set[str] = set()
    for _, row in history.iterrows():
        score = row.get("objective_score")
        status = str(row.get("status"))
        viable = bool(row.get("viable"))
        if score is None or pd.isna(score):
            continue
        if status in {"invalid", "error", "no_trades"} or (not viable and float(score) <= median_score):
            config = row.get("config")
            if not isinstance(config, dict):
                config = {key.removeprefix("param__"): value for key, value in row.items() if key.startswith("param__")}
            poor.add(coarse_signature_key(family, config))
    return poor


def _history_records(history: pd.DataFrame) -> list[dict[str, Any]]:
    if history.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in history.iterrows():
        record = row.to_dict()
        config = record.get("config")
        if not isinstance(config, dict):
            config = {key.removeprefix("param__"): value for key, value in record.items() if key.startswith("param__")}
        record["config"] = config
        records.append(record)
    return records


def _signature_maps(history: pd.DataFrame, family: str) -> tuple[set[str], dict[str, str]]:
    exact_hashes: set[str] = set()
    signature_map: dict[str, str] = {}
    if history.empty:
        return exact_hashes, signature_map

    for _, row in history.iterrows():
        config = row.get("config")
        if not isinstance(config, dict):
            config = {key.removeprefix("param__"): value for key, value in row.items() if key.startswith("param__")}
        config_hash = str(row.get("config_hash"))
        exact_hashes.add(config_hash)
        signature_map[config_hash] = coarse_signature_key(family, config)
    return exact_hashes, signature_map


def _template_counts(history: pd.DataFrame) -> dict[str, int]:
    if history.empty or "template_id" not in history.columns:
        return {}
    counts = history["template_id"].fillna("unspecified").astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _family_strategy_type(family: str) -> str:
    if family in {"ml_ranker"}:
        return "ml"
    if family in {"rl_bandit"}:
        return "rl"
    return "classical"


def _stagnation_batches(history: pd.DataFrame, lookback: int) -> int:
    if history.empty or "timestamp_utc" not in history.columns:
        return 0
    ranked = history.sort_values("timestamp_utc")
    best = float("-inf")
    stagnant = 0
    for _, row in ranked.iterrows():
        score = row.get("objective_score")
        if score is None or pd.isna(score):
            continue
        score = float(score)
        if score > best:
            best = score
            stagnant = 0
        else:
            stagnant += 1
    return max(0, min(stagnant, lookback))


def identify_dead_zones(history: pd.DataFrame, family: str) -> dict[str, set[Any]]:
    if history.empty:
        return {}
    dead_zones: dict[str, set[Any]] = {}
    median_score = float(history["objective_score"].fillna(float("-inf")).median()) if "objective_score" in history.columns else float("-inf")
    for column in [col for col in history.columns if col.startswith("param__")]:
        param = column.removeprefix("param__")
        bad_values: set[Any] = set()
        for value, group in history.groupby(column, dropna=False):
            if len(group) < 2:
                continue
            statuses = set(group["status"].astype(str))
            if statuses.issubset({"invalid", "error", "no_trades"}):
                bad_values.add(value)
                continue
            viable_rate = float(group["viable"].fillna(False).mean())
            avg_score = float(group["objective_score"].fillna(-999).mean())
            if viable_rate == 0.0 and avg_score <= median_score:
                bad_values.add(value)
                continue
            if len(group) >= 3 and avg_score < median_score:
                bad_values.add(value)
        if bad_values:
            dead_zones[param] = bad_values
    return dead_zones


def select_top_configs(history: pd.DataFrame, family: str, limit: int) -> pd.DataFrame:
    if history.empty:
        return history
    ranked = history.copy()
    ranked["viable_sort"] = ranked["viable"].fillna(False).astype(int)
    ranked["objective_sort"] = ranked["objective_score"].fillna(float("-inf"))
    ranked["status_penalty"] = ranked["status"].isin(["invalid", "error", "duplicate"]).astype(int)
    ranked = ranked.sort_values(
        ["viable_sort", "objective_sort", "status_penalty"],
        ascending=[False, False, True],
    )
    return ranked.head(limit).reset_index(drop=True)


def _neighbor_choices(values: list[Any], current: Any) -> list[Any]:
    if current not in values:
        return [value for value in values if value != current]
    idx = values.index(current)
    ordered: list[Any] = []
    if idx - 1 >= 0:
        ordered.append(values[idx - 1])
    if idx + 1 < len(values):
        ordered.append(values[idx + 1])
    ordered.extend(value for value in values if value not in ordered and value != current)
    return ordered


def build_local_neighbors(
    family: str,
    config: dict[str, Any],
    limit: int,
    seed: int,
    *,
    dead_zones: dict[str, set[Any]] | None = None,
    explored_hashes: set[str] | None = None,
) -> list[dict[str, Any]]:
    dead_zones = dead_zones or {}
    explored_hashes = explored_hashes or set()
    space = get_family_search_space(family)
    rng = random.Random(seed)
    keys = list(space.keys())
    rng.shuffle(keys)
    neighbors: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_candidate(candidate: dict[str, Any]) -> None:
        normalized = normalize_experiment_config(family, candidate)
        config_hash = compute_config_hash(family, normalized)
        if config_hash in explored_hashes or config_hash in seen:
            return
        seen.add(config_hash)
        neighbors.append(normalized)

    for key in keys:
        if len(neighbors) >= limit:
            break
        current = config.get(key)
        choices = _neighbor_choices(list(space[key]["choices"]), current)
        for choice in choices:
            candidate = dict(config)
            candidate[key] = choice
            add_candidate(candidate)
            if len(neighbors) >= limit:
                break

    if len(neighbors) < limit:
        for idx, key_a in enumerate(keys):
            for key_b in keys[idx + 1:]:
                for choice_a in _neighbor_choices(list(space[key_a]["choices"]), config.get(key_a)):
                    for choice_b in _neighbor_choices(list(space[key_b]["choices"]), config.get(key_b)):
                        candidate = dict(config)
                        candidate[key_a] = choice_a
                        candidate[key_b] = choice_b
                        add_candidate(candidate)
                        if len(neighbors) >= limit:
                            break
                    if len(neighbors) >= limit:
                        break
                if len(neighbors) >= limit:
                    break
            if len(neighbors) >= limit:
                break

    return neighbors[:limit]


def sample_exploration_configs(
    family: str,
    limit: int,
    seed: int,
    explored_hashes: set[str],
    *,
    dead_zones: dict[str, set[Any]] | None = None,
    sample_multiplier: int = 8,
) -> list[dict[str, Any]]:
    dead_zones = dead_zones or {}
    samples = sample_random_candidates(family, n=max(limit * sample_multiplier, limit * 2, 16), seed=seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in samples:
        config_hash = compute_config_hash(family, candidate)
        if config_hash in explored_hashes or config_hash in seen:
            continue
        seen.add(config_hash)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates


def analyze_experiment_history(
    *,
    families: list[str],
    base_dir: str = "experiments",
    baseline_by_family: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    baseline_by_family = {**DEFAULT_BASELINES, **(baseline_by_family or {})}
    memory = load_research_memory(base_dir)
    summary: dict[str, Any] = {"families": {}}
    for family in families:
        history = _load_detailed_history(family, base_dir)
        dead_zones = identify_dead_zones(history, family)
        median_score = float(history["objective_score"].fillna(float("-inf")).median()) if not history.empty else float("-inf")
        top = select_top_configs(history, family, limit=5)
        robust = top[top["viable"].fillna(False)] if not top.empty else top
        unstable = history[
            (history["objective_score"].fillna(float("-inf")) > history["objective_score"].fillna(float("-inf")).median())
            & (~history["viable"].fillna(False))
        ] if not history.empty else history
        exact_baseline = history[history["comparison_status"] == "exact_verified_current_engine"] if not history.empty else history

        exact_hashes, signature_map = _signature_maps(history, family)
        family_memory = (memory.get("families", {}) or {}).get(family, {})
        dead_zone_signatures = set(family_memory.get("dead_zone_signatures", []))
        poor_region_signatures = _poor_region_signatures(history, family, median_score)
        template_counts = _template_counts(history)
        stagnation = _stagnation_batches(history, lookback=10)
        summary["families"][family] = {
            "baseline_name": baseline_by_family.get(family),
            "history_count": int(len(history)),
            "success_count": int((history["status"] == "success").sum()) if not history.empty else 0,
            "viable_rate": float(history["viable"].fillna(False).mean()) if not history.empty else 0.0,
            "objective_mean": float(history["objective_score"].fillna(0).mean()) if not history.empty else 0.0,
            "objective_max": None if history.empty else float(history["objective_score"].fillna(float("-inf")).max()),
            "baseline_win_rate": (
                float(exact_baseline["beats_baseline_objective"].fillna(False).mean()) if not exact_baseline.empty else None
            ),
            "top_performers": top[["experiment_id", "config_hash", "objective_score", "viable"]].to_dict("records")
            if not top.empty else [],
            "robust_performers": robust[["experiment_id", "config_hash", "objective_score", "viable"]].to_dict("records")
            if not robust.empty else [],
            "overfit_or_unstable": unstable[["experiment_id", "config_hash", "objective_score", "viable"]].to_dict("records")
            if not unstable.empty else [],
            "dead_zones": {key: sorted(str(value) for value in values) for key, values in dead_zones.items()},
            "parameter_heuristics": _value_summary(history),
            "under_tested_regions": _under_tested_regions(_value_summary(history)),
            "explored_hashes": exact_hashes,
            "explored_signatures": signature_map,
            "dead_zone_signatures": dead_zone_signatures,
            "poor_region_signatures": poor_region_signatures,
            "template_counts": template_counts,
            "stagnation_batches": stagnation,
            "memory": family_memory,
            "history_frame": history,
        }
    return summary


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_family_search_space_size(family: str) -> int:
    size = 1
    for spec in get_family_search_space(family).values():
        size *= max(1, len(spec.get("choices") or []))
    return int(size)


def _normalize_budget_total(
    budgets: dict[str, int],
    *,
    total: int,
    weights: dict[str, float] | None = None,
) -> dict[str, int]:
    if total <= 0 or not budgets:
        return {}
    weights = weights or {family: 1.0 for family in budgets}
    normalized = {family: max(0, int(count)) for family, count in budgets.items()}

    if sum(normalized.values()) == 0:
        ordered = sorted(normalized, key=lambda family: (-weights.get(family, 0.0), family))
        for family in ordered[:total]:
            normalized[family] = 1

    while sum(normalized.values()) < total:
        family = max(normalized, key=lambda item: (weights.get(item, 0.0), -normalized[item], item))
        normalized[family] += 1

    while sum(normalized.values()) > total:
        removable = [
            family
            for family, count in normalized.items()
            if count > 0 and (count > 1 or sum(1 for value in normalized.values() if value > 0) > total)
        ]
        if not removable:
            break
        family = min(removable, key=lambda item: (weights.get(item, 0.0), -normalized[item], item))
        normalized[family] -= 1

    return {family: count for family, count in normalized.items() if count > 0}


def negotiate_family_budgets(
    request: ProposalRequest,
    analysis: dict[str, Any],
    guidance: dict[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate proposal budget without starving exploratory families."""
    families = list(request.strategy_families)
    total = int(request.max_experiments)
    if request.per_family_budgets:
        raw = {family: int(request.per_family_budgets.get(family, 0)) for family in families}
        budgets = _normalize_budget_total(raw, total=total)
        return budgets, {
            "mode": "explicit",
            "raw_budgets": raw,
            "normalized_budgets": budgets,
            "reason": "user-provided per_family_budgets normalized to requested total",
        }

    if len(families) == 1:
        budgets = {families[0]: total}
        return budgets, {"mode": "single_family", "weights": {families[0]: 1.0}, "normalized_budgets": budgets}

    base_weights = {
        "momentum": 0.60,
        "superstock": 0.25,
        "ml_ranker": 0.08,
        "rl_bandit": 0.07,
    }
    family_scores: dict[str, float] = {}
    details: dict[str, dict[str, Any]] = {}
    for family in families:
        family_analysis = analysis["families"].get(family, {})
        viable_rate = _safe_float(family_analysis.get("viable_rate"), 0.0)
        objective_max = _safe_float(family_analysis.get("objective_max"), 0.0)
        objective_mean = _safe_float(family_analysis.get("objective_mean"), 0.0)
        history_count = int(family_analysis.get("history_count") or 0)
        score = base_weights.get(family, 0.05)
        score += min(max(viable_rate, 0.0), 1.0) * 0.20
        score += max(min(objective_max, 3.0), 0.0) * 0.04
        score += max(min(objective_mean, 2.0), -2.0) * 0.02

        if family in {"ml_ranker", "rl_bandit"} and history_count >= 5 and viable_rate <= 0.0:
            score *= 0.45
        if family == "momentum" and viable_rate > 0.05:
            score *= 1.15
        if family == "superstock" and viable_rate > 0.10:
            score *= 1.05

        if guidance:
            for item in guidance.get("next_focus", []):
                if item.get("family") != family:
                    continue
                if item.get("focus") == "refine":
                    score *= 1.20
                elif item.get("focus") == "explore":
                    score *= 1.10
                elif item.get("focus") in {"deprioritize", "hold"}:
                    score *= 0.70

        family_scores[family] = max(score, 0.01)
        details[family] = {
            "base_weight": base_weights.get(family, 0.05),
            "viable_rate": viable_rate,
            "objective_max": objective_max,
            "objective_mean": objective_mean,
            "history_count": history_count,
            "negotiated_weight": round(family_scores[family], 6),
        }

    total_weight = sum(family_scores.values()) or float(len(families))
    raw = {
        family: int(round(total * family_scores[family] / total_weight))
        for family in families
    }
    min_alive = 1 if total >= len(families) else 0
    for family in families:
        if raw.get(family, 0) < min_alive:
            raw[family] = min_alive
    budgets = _normalize_budget_total(raw, total=total, weights=family_scores)
    return budgets, {
        "mode": "evidence_weighted",
        "family_scores": details,
        "raw_budgets": raw,
        "normalized_budgets": budgets,
        "reason": "weighted by viability, objective history, analysis guidance, and exploratory family floors",
    }


def _allocate_family_budgets(
    request: ProposalRequest,
    analysis: dict[str, Any],
    guidance: dict[str, Any] | None = None,
) -> dict[str, int]:
    budgets, _ = negotiate_family_budgets(request, analysis, guidance=guidance)
    return budgets


def minimum_viable_candidate_count(request: ProposalRequest) -> int:
    if request.min_viable_candidates is not None:
        return min(int(request.max_experiments), int(request.min_viable_candidates))
    fill_min = int(round(request.max_experiments * request.min_viable_fill_rate))
    if request.max_experiments >= request.large_search_threshold:
        fill_min = max(fill_min, min(request.max_experiments, request.min_large_search_candidates))
    return max(1, min(request.max_experiments, fill_min))


def _proposal_shortfall_reasons(
    *,
    request: ProposalRequest,
    family: str,
    budget: int,
    selected: int,
    family_analysis: dict[str, Any],
    metadata: list[dict[str, Any]],
) -> list[str]:
    if selected >= budget:
        return []

    reasons: list[str] = []
    search_space_size = _estimate_family_search_space_size(family)
    explored_count = len(family_analysis.get("explored_hashes") or [])
    dead_zone_count = len(family_analysis.get("dead_zones") or {})
    poor_signature_count = len(family_analysis.get("poor_region_signatures") or set())
    near_count = sum(1 for item in metadata if item.get("duplicate_risk") == "near")
    dead_zone_selected = sum(1 for item in metadata if _safe_float(item.get("dead_zone_risk"), 0.0) >= 0.5)

    if selected == 0 and budget > 0:
        reasons.append("zero_selected_for_family")
    if search_space_size <= explored_count + selected:
        reasons.append("exhausted_search_space")
    elif explored_count >= int(search_space_size * 0.80):
        reasons.append("duplicate_saturation")
    if near_count or request.max_near_duplicate_distance > 0:
        reasons.append("near_duplicate_suppression")
    if dead_zone_count or poor_signature_count or dead_zone_selected:
        reasons.append("dead_zone_suppression")
    if request.novelty_floor > 0.05:
        reasons.append("novelty_floor_too_strict")
    if budget > search_space_size:
        reasons.append("family_budget_overconstraint")
    if not reasons:
        reasons.append("candidate_generation_underfilled")
    return sorted(set(reasons))


def score_proposal_quality(
    request: ProposalRequest,
    *,
    budgets: dict[str, int],
    candidate_configs: dict[str, list[dict[str, Any]]],
    candidate_metadata: dict[str, list[dict[str, Any]]] | None,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    candidate_metadata = candidate_metadata or {}
    requested = int(request.max_experiments)
    candidate_count = sum(len(items) for items in candidate_configs.values())
    min_viable = minimum_viable_candidate_count(request)
    fill_rate = candidate_count / requested if requested else 0.0
    families_present = [family for family, configs in candidate_configs.items() if configs]

    family_reports: dict[str, Any] = {}
    shortfall_reasons: set[str] = set()
    source_types: set[str] = set()
    strategy_types: set[str] = set()
    duplicate_risk_count = 0
    dead_zone_heavy_count = 0
    local_refinement_count = 0
    exploration_count = 0

    for family in request.strategy_families:
        configs = candidate_configs.get(family, [])
        metadata = candidate_metadata.get(family, [])
        budget = int(budgets.get(family, 0))
        family_analysis = analysis["families"].get(family, {})
        reasons = _proposal_shortfall_reasons(
            request=request,
            family=family,
            budget=budget,
            selected=len(configs),
            family_analysis=family_analysis,
            metadata=metadata,
        )
        shortfall_reasons.update(reasons)
        family_source_types = sorted({str(item.get("source_type")) for item in metadata if item.get("source_type")})
        family_strategy_types = sorted({str(item.get("strategy_type")) for item in metadata if item.get("strategy_type")})
        source_types.update(family_source_types)
        strategy_types.update(family_strategy_types)
        duplicate_risk_count += sum(1 for item in metadata if item.get("duplicate_risk"))
        dead_zone_heavy_count += sum(1 for item in metadata if _safe_float(item.get("dead_zone_risk"), 0.0) >= 0.5)
        local_refinement_count += sum(1 for item in metadata if item.get("proposal_role") == "exploit")
        exploration_count += sum(1 for item in metadata if item.get("proposal_role") == "explore")
        family_reports[family] = {
            "budget": budget,
            "selected": len(configs),
            "fill_rate": (len(configs) / budget) if budget else 0.0,
            "search_space_size_estimate": _estimate_family_search_space_size(family),
            "explored_count": len(family_analysis.get("explored_hashes") or []),
            "source_types": family_source_types,
            "strategy_types": family_strategy_types,
            "shortfall_reasons": reasons,
        }

    diversity_score = min(1.0, (len(source_types) / 3.0) + (len(strategy_types) / 6.0))
    family_coverage = len(families_present) / max(1, len([family for family, value in budgets.items() if value > 0]))
    duplicate_penalty = min(0.35, duplicate_risk_count / max(1, candidate_count) * 0.35)
    dead_zone_penalty = min(0.25, dead_zone_heavy_count / max(1, candidate_count) * 0.25)
    size_score = min(1.0, fill_rate)
    quality_score = max(
        0.0,
        min(
            1.0,
            (0.55 * size_score)
            + (0.20 * family_coverage)
            + (0.15 * diversity_score)
            + (0.10 * (exploration_count / max(1, candidate_count)))
            - duplicate_penalty
            - dead_zone_penalty,
        ),
    )

    exhausted = bool(shortfall_reasons) and shortfall_reasons.issubset({"exhausted_search_space", "family_budget_overconstraint"})
    pass_minimum = candidate_count >= min_viable or exhausted
    large_search_underfilled = request.max_experiments >= request.large_search_threshold and candidate_count < min_viable and not exhausted
    quality_flags: list[str] = []
    if candidate_count < min_viable:
        quality_flags.append("too_small")
    if large_search_underfilled:
        quality_flags.append("large_search_underfilled")
    if candidate_count > 0 and len(source_types) <= 1 and len(strategy_types) <= 1 and len(request.strategy_families) > 1:
        quality_flags.append("too_narrow")
    if duplicate_risk_count / max(1, candidate_count) >= 0.25:
        quality_flags.append("duplicate_heavy")
    if dead_zone_heavy_count / max(1, candidate_count) >= 0.25:
        quality_flags.append("dead_zone_heavy")
    for family, report in family_reports.items():
        if report["budget"] > 0 and report["selected"] == 0:
            quality_flags.append(f"family_underfilled:{family}")
    heavy_quality_flags = {"duplicate_heavy", "dead_zone_heavy", "too_narrow"}
    if pass_minimum and quality_score >= 0.50 and not (heavy_quality_flags & set(quality_flags)):
        status = "pass"
    elif pass_minimum:
        status = "warn"
    else:
        status = "fail"

    return {
        "status": status,
        "quality_score": round(quality_score, 6),
        "requested": requested,
        "candidate_count": candidate_count,
        "min_viable_candidates": min_viable,
        "fill_rate": round(fill_rate, 6),
        "pass_minimum_viable_batch": pass_minimum,
        "large_search_underfilled": large_search_underfilled,
        "quality_flags": sorted(set(quality_flags)),
        "families_present": families_present,
        "family_reports": family_reports,
        "shortfall_reasons": sorted(shortfall_reasons),
        "duplicate_risk_count": duplicate_risk_count,
        "dead_zone_heavy_count": dead_zone_heavy_count,
        "local_refinement_count": local_refinement_count,
        "exploration_count": exploration_count,
        "source_type_count": len(source_types),
        "strategy_type_count": len(strategy_types),
        "execution_allowed": bool((not request.quality_gate) or pass_minimum),
    }


def _candidate_metadata(
    *,
    family: str,
    candidate: dict[str, Any],
    novelty: Any,
    strategy_type: str,
    source_type: str,
    template_id: str | None,
    hypothesis: str | None,
    reason_selected: str | None,
    exploration_mode: str,
    proposal_role: str,
    region_label: str | None,
    parent_config_hash: str | None,
    source_batch_ids: list[str],
    source_proposal_id: str | None,
    source_idea_ids: list[str] | None = None,
    template_tags: list[str] | None = None,
) -> dict[str, Any]:
    from experiment_store import compute_config_hash

    normalized = normalize_experiment_config(family, candidate)
    config_hash = compute_config_hash(family, normalized)
    return {
        "config": normalized,
        "config_hash": config_hash,
        "strategy_type": strategy_type,
        "source_type": source_type,
        "template_id": template_id,
        "hypothesis": hypothesis,
        "reason_selected": reason_selected,
        "novelty_score": getattr(novelty, "novelty_score", None),
        "selection_score": getattr(novelty, "selection_score", None),
        "objective_proxy": getattr(novelty, "objective_proxy", None),
        "exploration_mode": exploration_mode,
        "proposal_role": proposal_role,
        "region_label": region_label,
        "duplicate_risk": getattr(novelty, "duplicate_risk", None),
        "dead_zone_risk": getattr(novelty, "dead_zone_risk", None),
        "parent_config_hash": parent_config_hash,
        "near_duplicate_of": getattr(novelty, "near_duplicate_of", None),
        "dead_zone_flags": getattr(novelty, "dead_zone_flags", []) or [],
        "source_proposal_id": source_proposal_id,
        "source_batch_id": ",".join(source_batch_ids) if source_batch_ids else None,
        "source_idea_ids": list(source_idea_ids or []),
        "template_tags": template_tags or [],
    }


def _template_payloads_for_family(
    family: str,
    *,
    limit: int,
    seed: int,
    allow_cross_family: bool,
) -> list[dict[str, Any]]:
    payloads = expand_template_candidates(
        family,
        limit=limit,
        seed=seed,
        allow_cross_family=allow_cross_family,
    )
    if limit <= 0:
        return []
    return payloads[:limit]


def generate_next_round_proposal(
    request: ProposalRequest,
    *,
    base_dir: str = "experiments",
    baseline_by_family: dict[str, str | None] | None = None,
) -> ProposalResult:
    helper_ideas = _load_helper_ideas(request=request, base_dir=base_dir)
    analysis_guidance = _latest_analysis_guidance(request=request, base_dir=base_dir)
    request_source_idea_ids = list(dict.fromkeys((request.source_idea_ids or []) + [record.get("idea_id") for record in helper_ideas]))
    if request_source_idea_ids != list(request.source_idea_ids or []):
        request = ProposalRequest(
            proposal_id=request.proposal_id,
            timestamp_utc=request.timestamp_utc,
            source_batch_ids=request.source_batch_ids,
            strategy_families=request.strategy_families,
            objective_name=request.objective_name,
            baseline_name=request.baseline_name,
            seed=request.seed,
            exploration_fraction=request.exploration_fraction,
            exploitation_fraction=request.exploitation_fraction,
            max_experiments=request.max_experiments,
            per_family_budgets=request.per_family_budgets,
            resume=request.resume,
            novelty_floor=request.novelty_floor,
            template_fraction=request.template_fraction,
            cross_family_fraction=request.cross_family_fraction,
            max_near_duplicate_distance=request.max_near_duplicate_distance,
            stagnation_escape_batches=request.stagnation_escape_batches,
            allow_external_seeds=request.allow_external_seeds,
            source_idea_ids=request_source_idea_ids,
            use_idea_queue=request.use_idea_queue,
            use_analysis_guidance=request.use_analysis_guidance,
            min_viable_fill_rate=request.min_viable_fill_rate,
            min_viable_candidates=request.min_viable_candidates,
            large_search_threshold=request.large_search_threshold,
            min_large_search_candidates=request.min_large_search_candidates,
            quality_gate=request.quality_gate,
        )
    analysis = analyze_experiment_history(
        families=request.strategy_families,
        base_dir=base_dir,
        baseline_by_family=baseline_by_family,
    )
    all_source_batch_ids = sorted(
        {
            str(batch_id)
            for family in request.strategy_families
            for batch_id in analysis["families"][family]["history_frame"].get("batch_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        }
    )
    if not request.source_batch_ids:
        request = ProposalRequest(
            proposal_id=request.proposal_id,
            timestamp_utc=request.timestamp_utc,
            source_batch_ids=all_source_batch_ids,
            strategy_families=request.strategy_families,
            objective_name=request.objective_name,
            baseline_name=request.baseline_name,
            seed=request.seed,
            exploration_fraction=request.exploration_fraction,
            exploitation_fraction=request.exploitation_fraction,
            max_experiments=request.max_experiments,
            per_family_budgets=request.per_family_budgets,
            resume=request.resume,
            novelty_floor=request.novelty_floor,
            template_fraction=request.template_fraction,
            cross_family_fraction=request.cross_family_fraction,
            max_near_duplicate_distance=request.max_near_duplicate_distance,
            stagnation_escape_batches=request.stagnation_escape_batches,
            allow_external_seeds=request.allow_external_seeds,
            source_idea_ids=request.source_idea_ids,
            use_idea_queue=request.use_idea_queue,
            use_analysis_guidance=request.use_analysis_guidance,
            min_viable_fill_rate=request.min_viable_fill_rate,
            min_viable_candidates=request.min_viable_candidates,
            large_search_threshold=request.large_search_threshold,
            min_large_search_candidates=request.min_large_search_candidates,
            quality_gate=request.quality_gate,
        )
    budgets, budget_negotiation = negotiate_family_budgets(request, analysis, guidance=analysis_guidance)
    candidate_configs: dict[str, list[dict[str, Any]]] = {}
    candidate_metadata: dict[str, list[dict[str, Any]]] = {}
    reasoning: dict[str, Any] = {
        "source_batch_ids": request.source_batch_ids or all_source_batch_ids,
        "source_idea_ids": request.source_idea_ids or [],
        "family_budget_decision": budgets,
        "family_budget_negotiation": budget_negotiation,
        "analysis_guidance": analysis_guidance.get("next_focus") if analysis_guidance else [],
        "novelty_policy": {
            "exploration_fraction": request.exploration_fraction,
            "exploitation_fraction": request.exploitation_fraction,
            "novelty_floor": request.novelty_floor,
            "template_fraction": request.template_fraction,
            "cross_family_fraction": request.cross_family_fraction,
            "max_near_duplicate_distance": request.max_near_duplicate_distance,
            "stagnation_escape_batches": request.stagnation_escape_batches,
            "allow_external_seeds": request.allow_external_seeds,
        },
        "families": {},
    }
    memory = load_research_memory(base_dir)

    for idx, family in enumerate(request.strategy_families):
        family_analysis = analysis["families"][family]
        history = family_analysis["history_frame"]
        history_records = _history_records(history)
        explored_hashes = set(family_analysis["explored_hashes"])
        explored_signatures = dict(family_analysis["explored_signatures"])
        dead_zones = identify_dead_zones(history, family)
        family_memory = (memory.get("families", {}) or {}).get(family, {})
        dead_zone_values = {
            param: {value for value in values}
            for param, values in dead_zones.items()
        }
        for param, values in (family_memory.get("dead_zone_values") or {}).items():
            dead_zone_values.setdefault(param, set()).update(values)
        dead_zone_signatures = set(family_analysis.get("dead_zone_signatures", set()))
        dead_zone_signatures.update(family_analysis.get("poor_region_signatures", set()))
        budget = int(budgets.get(family, 0))
        stagnation_batches = max(int(family_analysis.get("stagnation_batches") or 0), int(family_memory.get("stagnation_batches") or 0))
        exploration_fraction = request.exploration_fraction
        exploitation_fraction = request.exploitation_fraction
        novelty_floor = request.novelty_floor
        if stagnation_batches >= request.stagnation_escape_batches:
            exploration_fraction = max(exploration_fraction, 0.80)
            exploitation_fraction = 1.0 - exploration_fraction
            novelty_floor = max(0.05, novelty_floor * 0.75)
            budget = min(max(1, int(round(budget * 1.25))), request.max_experiments)

        exploit_n = max(1, min(budget, int(round(budget * exploitation_fraction)))) if budget > 0 else 0
        explore_n = max(0, budget - exploit_n)
        template_n = max(1, int(round(explore_n * request.template_fraction))) if explore_n else 0
        cross_family_n = 0
        if len(request.strategy_families) > 1 and explore_n:
            cross_family_n = max(1, int(round(explore_n * request.cross_family_fraction)))
            cross_family_n = min(cross_family_n, explore_n)
        local_explore_n = max(0, explore_n - template_n - cross_family_n)

        top = select_top_configs(history, family, limit=max(1, min(8, exploit_n or 1)))
        seen_hashes = set(explored_hashes)
        seen_signatures = set(explored_signatures.values())
        selection_pool: list[dict[str, Any]] = []
        family_ideas = [record for record in helper_ideas if record.get("family") in {None, family}]
        guidance_focus = None
        if analysis_guidance:
            for item in analysis_guidance.get("next_focus", []):
                if item.get("family") == family:
                    guidance_focus = item
                    break

        def _try_add_candidate(
            candidate: dict[str, Any],
            *,
            strategy_type: str,
            source_type: str,
            template_id: str | None,
            hypothesis: str | None,
            reason_selected: str | None,
            exploration_mode: str,
            proposal_role: str,
            region_label: str | None,
            parent_config_hash: str | None = None,
            source_idea_ids: list[str] | None = None,
            template_tags: list[str] | None = None,
            allow_near_duplicate: bool = False,
        ) -> None:
            novelty = score_candidate(
                family,
                candidate,
                history=history_records,
                explored_hashes=seen_hashes,
                explored_signatures=explored_signatures,
                dead_zone_values=dead_zone_values,
                dead_zone_signatures=dead_zone_signatures,
                source_type=source_type,
                template_id=template_id,
                exploration_mode=exploration_mode,
                novelty_floor=novelty_floor,
                near_duplicate_distance=request.max_near_duplicate_distance,
            )
            if novelty is None:
                return
            if novelty.near_duplicate and not allow_near_duplicate:
                return
            normalized = normalize_experiment_config(family, candidate)
            config_hash = compute_config_hash(family, normalized)
            if config_hash in seen_hashes:
                return
            signature = coarse_signature_key(family, normalized)
            if signature in seen_signatures and not allow_near_duplicate:
                return
            if any(
                signature_distance(family, normalized, existing["config"]) <= request.max_near_duplicate_distance
                for existing in selection_pool
            ) and not allow_near_duplicate:
                return
            seen_hashes.add(config_hash)
            seen_signatures.add(signature)
            selection_pool.append(
                _candidate_metadata(
                    family=family,
                    candidate=normalized,
                    novelty=novelty,
                    strategy_type=strategy_type,
                    source_type=source_type,
                    template_id=template_id,
                    hypothesis=hypothesis,
                    reason_selected=reason_selected,
                    exploration_mode=exploration_mode,
                    proposal_role=proposal_role,
                    region_label=region_label,
                    parent_config_hash=parent_config_hash,
                    source_batch_ids=request.source_batch_ids or all_source_batch_ids,
                    source_proposal_id=request.proposal_id,
                    source_idea_ids=source_idea_ids,
                    template_tags=template_tags,
                )
            )

        for idea in family_ideas:
            if len(selection_pool) >= budget:
                break
            suggested = idea.get("suggested_config") or {}
            seed_config: dict[str, Any] | None = None
            parent_hash = suggested.get("parent_config_hash") if isinstance(suggested, dict) else None
            if parent_hash:
                match = history[history["config_hash"].astype(str) == str(parent_hash)] if not history.empty else history
                if not match.empty:
                    seed_config = dict(match.iloc[0].get("config") or {})
            if not seed_config and isinstance(suggested, dict):
                seed_config = {k: v for k, v in suggested.items() if k != "parent_config_hash"}
            if not seed_config:
                continue
            neighbors = build_local_neighbors(
                family,
                seed_config,
                limit=2,
                seed=request.seed + idx + len(selection_pool),
                dead_zones=dead_zones,
                explored_hashes=seen_hashes,
            )
            for neighbor in neighbors:
                if len(selection_pool) >= budget:
                    break
                _try_add_candidate(
                    neighbor,
                    source_type="idea_seed",
                    strategy_type=idea.get("strategy_type") or _family_strategy_type(family),
                    template_id=idea.get("suggested_template_id"),
                    hypothesis=idea.get("hypothesis"),
                    reason_selected=idea.get("rationale") or "seeded from helper idea queue",
                    exploration_mode="idea_seed",
                    proposal_role="explore",
                    region_label=idea.get("suggested_template_id") or "idea_seed",
                    parent_config_hash=parent_hash,
                    source_idea_ids=[idea.get("idea_id")] if idea.get("idea_id") else [],
                    allow_near_duplicate=True,
                )

        for top_idx, row in top.iterrows():
            if len(selection_pool) >= exploit_n:
                break
            config = row.get("config") or {}
            neighbors = build_local_neighbors(
                family,
                config,
                limit=max(1, exploit_n - len(selection_pool)),
                seed=request.seed + idx + top_idx,
                dead_zones=dead_zones,
                explored_hashes=seen_hashes,
            )
            for neighbor in neighbors:
                if len(selection_pool) >= exploit_n:
                    break
                _try_add_candidate(
                    neighbor,
                    source_type="local_refinement",
                    strategy_type=_family_strategy_type(family),
                    template_id=row.get("template_id"),
                    hypothesis=row.get("hypothesis") or family_analysis.get("baseline_name"),
                    reason_selected="local refinement around top historical configs",
                    exploration_mode="local_refinement",
                    proposal_role="exploit",
                    region_label=row.get("template_id") or "local_refinement",
                    parent_config_hash=row.get("config_hash"),
                    source_idea_ids=[],
                    allow_near_duplicate=False,
                )

        template_payloads = _template_payloads_for_family(
            family,
            limit=max(template_n + cross_family_n + 4, 8),
            seed=request.seed + 1000 + idx,
            allow_cross_family=True,
        )
        template_payloads.sort(key=lambda item: item["metadata"].get("source_type") != "cross_family_hybrid")
        template_count = 0
        cross_count = 0
        for payload in template_payloads:
            if len(selection_pool) >= budget:
                break
            metadata = payload["metadata"]
            if metadata.get("source_type") == "cross_family_hybrid":
                if cross_count >= cross_family_n and len(selection_pool) >= exploit_n + template_n:
                    continue
                cross_count += 1
            else:
                if template_count >= template_n and len(selection_pool) >= exploit_n + template_n:
                    continue
                template_count += 1
            _try_add_candidate(
                payload["config"],
                source_type=metadata.get("source_type") or "template_expansion",
                strategy_type=metadata.get("strategy_type") or _family_strategy_type(family),
                template_id=metadata.get("template_id"),
                hypothesis=metadata.get("hypothesis"),
                reason_selected=metadata.get("reason_selected"),
                exploration_mode=metadata.get("exploration_mode") or "template_expansion",
                proposal_role="explore",
                region_label=metadata.get("template_id") or metadata.get("source_type") or "template_expansion",
                parent_config_hash=family_analysis.get("top_performers", [{}])[0].get("config_hash") if family_analysis.get("top_performers") else None,
                source_idea_ids=[],
                template_tags=metadata.get("tags"),
                allow_near_duplicate=False,
            )

        exploration_candidates = sample_exploration_configs(
            family,
            limit=max(local_explore_n * 4, local_explore_n, 4),
            seed=request.seed + 2000 + idx,
            explored_hashes=seen_hashes,
            dead_zones=dead_zones,
        )
        for candidate in exploration_candidates:
            if len(selection_pool) >= budget:
                break
            _try_add_candidate(
                candidate,
                source_type="broad_exploration",
                strategy_type=_family_strategy_type(family),
                template_id="under_tested_region",
                hypothesis="Explore under-tested parameter regions.",
                reason_selected="coverage of under-tested regions and dead-zone escape",
                exploration_mode="broader_exploration",
                proposal_role="explore",
                region_label="under_tested_region",
                source_idea_ids=[],
                allow_near_duplicate=False,
            )

        if request.allow_external_seeds:
            for seed_payload in load_external_idea_seeds(enabled=True):
                if len(selection_pool) >= budget:
                    break
                _try_add_candidate(
                    seed_payload.get("config", {}),
                    source_type="external_seed",
                    strategy_type=seed_payload.get("strategy_type") or _family_strategy_type(family),
                    template_id=seed_payload.get("template_id"),
                    hypothesis=seed_payload.get("hypothesis"),
                    reason_selected=seed_payload.get("reason_selected"),
                    exploration_mode="external_seed",
                    proposal_role="explore",
                    region_label=seed_payload.get("template_id") or "external_seed",
                    parent_config_hash=seed_payload.get("parent_config_hash"),
                    source_idea_ids=[],
                    allow_near_duplicate=False,
                )

        if len(selection_pool) < budget:
            fallback = sample_exploration_configs(
                family,
                limit=budget - len(selection_pool),
                seed=request.seed + 3000 + idx,
                explored_hashes=seen_hashes,
                dead_zones={},
            )
            for candidate in fallback:
                if len(selection_pool) >= budget:
                    break
                _try_add_candidate(
                    candidate,
                    source_type="broad_exploration",
                    strategy_type=_family_strategy_type(family),
                    template_id="fallback_random",
                    hypothesis="Fallback random exploration when candidate pool is sparse.",
                    reason_selected="fill remaining budget with novel fallback samples",
                    exploration_mode="broader_exploration",
                    proposal_role="explore",
                    region_label="fallback_random",
                    source_idea_ids=[],
                    allow_near_duplicate=False,
                )

        if len(selection_pool) < budget:
            saturation_escape = sample_exploration_configs(
                family,
                limit=budget - len(selection_pool),
                seed=request.seed + 4000 + idx,
                explored_hashes=seen_hashes,
                dead_zones={},
                sample_multiplier=64,
            )
            for candidate in saturation_escape:
                if len(selection_pool) >= budget:
                    break
                _try_add_candidate(
                    candidate,
                    source_type="saturation_escape",
                    strategy_type=_family_strategy_type(family),
                    template_id="saturation_escape_random",
                    hypothesis="Saturation escape when normal novelty filters produce too few candidates.",
                    reason_selected="exact-new config selected after normal proposal filters were exhausted",
                    exploration_mode="saturation_escape",
                    proposal_role="explore",
                    region_label="saturation_escape",
                    source_idea_ids=[],
                    allow_near_duplicate=True,
                )

        ranked_payloads = sorted(
            selection_pool,
            key=lambda item: (
                -int(bool(item.get("source_idea_ids"))),
                -float(item.get("selection_score") or 0.0),
                -float(item.get("novelty_score") or 0.0),
                str(item.get("config_hash")),
            ),
        )
        family_source_idea_ids = [idea.get("idea_id") for idea in family_ideas if idea.get("idea_id")]
        if family_source_idea_ids:
            for item in ranked_payloads:
                if not item.get("source_idea_ids"):
                    item["source_idea_ids"] = list(family_source_idea_ids)
        candidate_configs[family] = [item["config"] for item in ranked_payloads[:budget]]
        candidate_metadata[family] = [
            {key: value for key, value in item.items() if key != "config"} for item in ranked_payloads[:budget]
        ]
        reasoning["families"][family] = {
            "baseline_name": family_analysis.get("baseline_name"),
            "history_count": family_analysis["history_count"],
            "top_performers": family_analysis["top_performers"][:3],
            "robust_performers": family_analysis["robust_performers"][:3],
            "overfit_or_unstable": family_analysis["overfit_or_unstable"][:3],
            "dead_zones": family_analysis["dead_zones"],
            "poor_region_signatures": sorted(family_analysis.get("poor_region_signatures", set())),
            "under_tested_regions": family_analysis["under_tested_regions"],
            "source_idea_ids": [idea.get("idea_id") for idea in family_ideas if idea.get("idea_id")],
            "analysis_next_focus": guidance_focus,
            "template_counts": family_analysis.get("template_counts", {}),
            "stagnation_batches": stagnation_batches,
            "exploit_count": min(len(candidate_configs[family]), exploit_n),
            "explore_count": max(0, len(candidate_configs[family]) - min(len(candidate_configs[family]), exploit_n)),
            "exploration_fraction": exploration_fraction,
            "exploitation_fraction": exploitation_fraction,
            "deprioritized_regions": family_analysis["dead_zones"],
            "baseline_win_rate": family_analysis["baseline_win_rate"],
            "novelty_floor": novelty_floor,
            "selection_count": len(candidate_configs[family]),
        }

    proposal_memory = load_research_memory(base_dir)
    for family in request.strategy_families:
        family_analysis = analysis["families"][family]
        history = family_analysis["history_frame"]
        exact_hashes = set(family_analysis["explored_hashes"])
        signatures = set(family_analysis["explored_signatures"].values())
        dead_zones = identify_dead_zones(history, family)
        family_memory = (proposal_memory.get("families", {}) or {}).get(family, {})
        template_counts = family_analysis.get("template_counts", {})
        stagnation_batches = max(int(family_analysis.get("stagnation_batches") or 0), int(family_memory.get("stagnation_batches") or 0))
        proposal_memory = update_family_memory(
            proposal_memory,
            family,
            exact_hashes=exact_hashes,
            coarse_signatures=signatures,
            dead_zone_values={key: {str(value) for value in values} for key, values in dead_zones.items()},
            dead_zone_signatures=set(family_analysis.get("dead_zone_signatures", set())),
            poor_region_signatures=set(family_analysis.get("poor_region_signatures", set())),
            template_counts=template_counts,
            best_objective_score=family_analysis.get("objective_max"),
            best_config_hash=family_analysis.get("top_performers", [{}])[0].get("config_hash") if family_analysis.get("top_performers") else None,
            stagnation_batches=stagnation_batches,
        )
    save_research_memory(proposal_memory, base_dir)

    reasoning["proposal_quality"] = score_proposal_quality(
        request,
        budgets=budgets,
        candidate_configs=candidate_configs,
        candidate_metadata=candidate_metadata,
        analysis=analysis,
    )

    proposal = ProposalResult(
        request=request,
        status="generated",
        candidate_configs=candidate_configs,
        candidate_metadata=candidate_metadata,
        reasoning_summary=reasoning,
    )
    payload = asdict(proposal)
    saved = save_proposal_result(payload, base_dir=base_dir)
    return ProposalResult(
        request=proposal.request,
        status=proposal.status,
        candidate_configs=proposal.candidate_configs,
        candidate_metadata=proposal.candidate_metadata,
        reasoning_summary=proposal.reasoning_summary,
        proposal_path=saved["proposal_path"],
        summary_path=saved["summary_path"],
    )
