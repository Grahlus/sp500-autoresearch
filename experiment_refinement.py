from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_batch import DEFAULT_BASELINES
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
    exploration_fraction: float = 0.30,
    exploitation_fraction: float = 0.70,
    max_experiments: int = 20,
    per_family_budgets: dict[str, int] | None = None,
    resume: bool = True,
    proposal_id: str | None = None,
    timestamp_utc: str | None = None,
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
    )


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


def identify_dead_zones(history: pd.DataFrame, family: str) -> dict[str, set[Any]]:
    if history.empty:
        return {}
    dead_zones: dict[str, set[Any]] = {}
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
            if viable_rate == 0.0 and avg_score < 0.0:
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
        for param, bad_values in dead_zones.items():
            if normalized.get(param) in bad_values:
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
) -> list[dict[str, Any]]:
    dead_zones = dead_zones or {}
    samples = sample_random_candidates(family, n=max(limit * 4, limit), seed=seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in samples:
        config_hash = compute_config_hash(family, candidate)
        if config_hash in explored_hashes or config_hash in seen:
            continue
        if any(candidate.get(param) in bad_values for param, bad_values in dead_zones.items()):
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
    summary: dict[str, Any] = {"families": {}}
    for family in families:
        history = _load_detailed_history(family, base_dir)
        dead_zones = identify_dead_zones(history, family)
        top = select_top_configs(history, family, limit=5)
        robust = top[top["viable"].fillna(False)] if not top.empty else top
        unstable = history[
            (history["objective_score"].fillna(float("-inf")) > history["objective_score"].fillna(float("-inf")).median())
            & (~history["viable"].fillna(False))
        ] if not history.empty else history
        exact_baseline = history[history["comparison_status"] == "exact_verified_current_engine"] if not history.empty else history

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
            "explored_hashes": set(history["config_hash"].astype(str)) if not history.empty else set(),
            "history_frame": history,
        }
    return summary


def _allocate_family_budgets(
    request: ProposalRequest,
    analysis: dict[str, Any],
) -> dict[str, int]:
    if request.per_family_budgets:
        return {family: int(count) for family, count in request.per_family_budgets.items()}
    families = request.strategy_families
    if len(families) == 1:
        return {families[0]: request.max_experiments}

    family_scores = []
    for family in families:
        details = analysis["families"][family]
        score = (
            float(details.get("viable_rate") or 0.0),
            float(details.get("objective_mean") or 0.0),
        )
        family_scores.append((family, score))
    family_scores.sort(key=lambda item: item[1], reverse=True)
    first, second = family_scores[0][0], family_scores[1][0]
    first_details = analysis["families"][first]
    second_details = analysis["families"][second]
    if first_details["viable_rate"] > second_details["viable_rate"] and first_details["history_count"] > 0:
        first_budget = max(int(round(request.max_experiments * 0.60)), 1)
        second_budget = max(request.max_experiments - first_budget, 1)
    else:
        first_budget = request.max_experiments // 2
        second_budget = request.max_experiments - first_budget
    budgets = {first: first_budget, second: second_budget}
    for family in families[2:]:
        budgets[family] = max(1, request.max_experiments // len(families))
    return budgets


def generate_next_round_proposal(
    request: ProposalRequest,
    *,
    base_dir: str = "experiments",
    baseline_by_family: dict[str, str | None] | None = None,
) -> ProposalResult:
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
        )
    budgets = _allocate_family_budgets(request, analysis)
    candidate_configs: dict[str, list[dict[str, Any]]] = {}
    reasoning: dict[str, Any] = {
        "source_batch_ids": request.source_batch_ids or all_source_batch_ids,
        "family_budget_decision": budgets,
        "families": {},
    }

    for idx, family in enumerate(request.strategy_families):
        family_analysis = analysis["families"][family]
        history = family_analysis["history_frame"]
        explored_hashes = set(family_analysis["explored_hashes"])
        dead_zones = identify_dead_zones(history, family)
        budget = int(budgets.get(family, 0))
        exploit_n = min(budget, max(0, int(round(budget * request.exploitation_fraction))))
        explore_n = max(0, budget - exploit_n)

        top = select_top_configs(history, family, limit=max(1, min(5, exploit_n or 1)))
        exploit_candidates: list[dict[str, Any]] = []
        for top_idx, row in top.iterrows():
            if len(exploit_candidates) >= exploit_n:
                break
            config = row.get("config") or {}
            neighbors = build_local_neighbors(
                family,
                config,
                limit=max(1, exploit_n - len(exploit_candidates)),
                seed=request.seed + idx + top_idx,
                dead_zones=dead_zones,
                explored_hashes=explored_hashes | {compute_config_hash(family, candidate) for candidate in exploit_candidates},
            )
            exploit_candidates.extend(neighbors)
        exploit_candidates = exploit_candidates[:exploit_n]

        explore_candidates = sample_exploration_configs(
            family,
            limit=explore_n,
            seed=request.seed + 100 + idx,
            explored_hashes=explored_hashes | {compute_config_hash(family, candidate) for candidate in exploit_candidates},
            dead_zones=dead_zones,
        )

        candidates = exploit_candidates + explore_candidates
        if len(candidates) < budget:
            fallback = sample_exploration_configs(
                family,
                limit=budget - len(candidates),
                seed=request.seed + 200 + idx,
                explored_hashes=explored_hashes | {compute_config_hash(family, candidate) for candidate in candidates},
                dead_zones={},
            )
            candidates.extend(fallback)

        candidate_configs[family] = candidates[:budget]
        reasoning["families"][family] = {
            "baseline_name": family_analysis.get("baseline_name"),
            "history_count": family_analysis["history_count"],
            "top_performers": family_analysis["top_performers"][:3],
            "robust_performers": family_analysis["robust_performers"][:3],
            "overfit_or_unstable": family_analysis["overfit_or_unstable"][:3],
            "dead_zones": family_analysis["dead_zones"],
            "under_tested_regions": family_analysis["under_tested_regions"],
            "exploit_count": len(exploit_candidates[:budget]),
            "explore_count": len(explore_candidates[:budget]),
            "deprioritized_regions": family_analysis["dead_zones"],
            "baseline_win_rate": family_analysis["baseline_win_rate"],
        }

    proposal = ProposalResult(
        request=request,
        status="generated",
        candidate_configs=candidate_configs,
        reasoning_summary=reasoning,
    )
    payload = asdict(proposal)
    saved = save_proposal_result(payload, base_dir=base_dir)
    return ProposalResult(
        request=proposal.request,
        status=proposal.status,
        candidate_configs=proposal.candidate_configs,
        reasoning_summary=proposal.reasoning_summary,
        proposal_path=saved["proposal_path"],
        summary_path=saved["summary_path"],
    )
