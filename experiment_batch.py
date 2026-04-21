from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_objective import rank_results
from experiment_memory_guard import current_process_memory_kb
from experiment_hot_index import compact_batch_summary_for_disk, refresh_hot_index_reports, upsert_batch_summary
from experiment_parallel import run_experiments_parallel
from experiment_runner import run_single_experiment
from experiment_spaces import enumerate_grid_candidates, list_searchable_families, sample_random_candidates
from experiment_store import compute_config_hash, init_store, load_prior_results, save_experiment_result_atomic
from experiment_types import BatchRequest, BatchResult, ExperimentResult, ExperimentSpec, ProposalResult
from prepare import load_data


DEFAULT_BASELINES = {
    "momentum": "momentum_champion_s10005",
}

BATCH_SUMMARY_DISK_LIMIT_BYTES = 2_000_000

DEFAULT_STRATEGY_TYPES = {
    "amihud_illiquidity_premium": "classical",
    "fear_greed_contrarian": "classical",
    "fear_greed_contrarian_overlay": "classical",
    "sector_breadth_overlay": "classical",
    "volatility_compression_expansion": "classical",
    "momentum": "classical",
    "superstock": "classical",
    "ml_ranker": "ml",
    "rl_bandit": "rl",
}


def build_batch_request(
    *,
    strategy_families: list[str],
    sampler_type: str = "random",
    max_experiments: int = 20,
    max_per_family: int | None = None,
    seed: int = 42,
    persist: bool = True,
    resume: bool = True,
    objective_name: str = "wf_v1_score",
    max_workers: int | None = None,
    fail_fast: bool = False,
    execution_mode: str = "auto",
    include_filters: dict[str, Any] | None = None,
    exclude_filters: dict[str, Any] | None = None,
    batch_id: str | None = None,
    timestamp_utc: str | None = None,
) -> BatchRequest:
    families = [family.strip().lower() for family in strategy_families if family and family.strip()]
    known = set(list_searchable_families())
    unknown = sorted(set(families) - known)
    if unknown:
        raise ValueError(f"Unknown strategy family(s): {', '.join(unknown)}")
    if sampler_type not in {"random", "grid"}:
        raise ValueError("sampler_type must be 'random' or 'grid'.")
    if max_experiments <= 0:
        raise ValueError("max_experiments must be > 0.")
    resolved_max_per_family = max_per_family if max_per_family is not None else max_experiments
    if resolved_max_per_family <= 0:
        raise ValueError("max_per_family must be > 0.")
    if max_workers is not None and max_workers > 8:
        raise ValueError("max_workers must be <= 8.")
    if max_workers is not None and max_workers <= 0:
        raise ValueError("max_workers must be > 0 when provided.")

    return BatchRequest(
        batch_id=batch_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        timestamp_utc=timestamp_utc or datetime.now(UTC).isoformat(),
        strategy_families=families,
        sampler_type=sampler_type,
        max_experiments=int(max_experiments),
        max_per_family=int(resolved_max_per_family),
        seed=int(seed),
        persist=bool(persist),
        resume=bool(resume),
        objective_name=objective_name,
        max_workers=max_workers,
        fail_fast=bool(fail_fast),
        execution_mode=execution_mode,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        precomputed_configs=None,
        source_proposal_id=None,
        proposal_metadata=None,
    )


def sample_batch_configs(
    family: str,
    method: str,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    if method == "grid":
        return enumerate_grid_candidates(family, limit=n)
    return sample_random_candidates(family, n=n, seed=seed)


def _result_to_row(result: ExperimentResult) -> dict[str, Any]:
    baseline = result.baseline_comparison or {}
    spec = result.spec
    return {
        "experiment_id": spec.experiment_id,
        "strategy_family": spec.family,
        "config_hash": spec.config_hash,
        "strategy_type": spec.strategy_type,
        "source_type": spec.source_type,
        "template_id": spec.template_id,
        "hypothesis": spec.hypothesis,
        "reason_selected": spec.reason_selected,
        "novelty_score": spec.novelty_score,
        "selection_score": spec.selection_score,
        "exploration_mode": spec.exploration_mode,
        "proposal_role": spec.proposal_role,
        "region_label": spec.region_label,
        "duplicate_risk": spec.duplicate_risk,
        "dead_zone_risk": spec.dead_zone_risk,
        "parent_config_hash": spec.parent_config_hash,
        "near_duplicate_of": spec.near_duplicate_of,
        "dead_zone_flags": spec.dead_zone_flags,
        "source_proposal_id": spec.source_proposal_id,
        "source_region_id": spec.source_region_id,
        "source_grid_search_id": spec.source_grid_search_id,
        "source_cells": spec.source_cells,
        "region_class": spec.region_class,
        "promotion_recommendation": spec.promotion_recommendation,
        "region_state": spec.region_state,
        "neighbor_of_cell": spec.neighbor_of_cell,
        "neighbor_generation_reason": spec.neighbor_generation_reason,
        "neighbor_grid_status": spec.neighbor_grid_status,
        "validator_mode": spec.validator_mode,
        "allowed_override_params": spec.allowed_override_params,
        "allowed_override_values": spec.allowed_override_values,
        "off_grid_params": spec.off_grid_params,
        "validation_override_reason": spec.validation_override_reason,
        "source_idea_ids": spec.source_idea_ids,
        "confirmation_state": spec.confirmation_state,
        "confirmation_required": spec.confirmation_required,
        "confirmation_reason": spec.confirmation_reason,
        "confirmation_batch_id": spec.confirmation_batch_id,
        "confirmation_trial_kind": spec.confirmation_trial_kind,
        "targeted_follow_up_required": spec.targeted_follow_up_required,
        "targeted_follow_up_reason": spec.targeted_follow_up_reason,
        "targeted_follow_up_type": spec.targeted_follow_up_type,
        "targeted_follow_up_priority": spec.targeted_follow_up_priority,
        "targeted_follow_up_batch_id": spec.targeted_follow_up_batch_id,
        "holdout_check_required": spec.holdout_check_required,
        "holdout_check_type": spec.holdout_check_type,
        "holdout_check_status": spec.holdout_check_status,
        "holdout_check_outcome": spec.holdout_check_outcome,
        "holdout_check_scope": spec.holdout_check_scope,
        "holdout_check_batch_id": spec.holdout_check_batch_id,
        "holdout_horizon_tags": spec.holdout_horizon_tags,
        "holdout_regime_tags": spec.holdout_regime_tags,
        "objective_score": result.objective_score,
        "sharpe": result.metrics.get("sharpe"),
        "calmar": result.metrics.get("calmar"),
        "total_return": result.metrics.get("total_return"),
        "max_drawdown": result.metrics.get("max_drawdown"),
        "trades_per_year": result.metrics.get("trades_per_year"),
        "exposure": result.metrics.get("exposure"),
        "status": result.status,
        "viable": result.robustness.get("viable"),
        "baseline_name": baseline.get("baseline_name"),
        "comparison_status": baseline.get("comparison_status"),
        "delta_sharpe": baseline.get("delta_sharpe"),
        "delta_calmar": baseline.get("delta_calmar"),
        "delta_return": baseline.get("delta_return"),
        "beats_baseline_objective": baseline.get("beats_baseline_objective"),
        "beats_baseline_guardrails": baseline.get("beats_baseline_guardrails"),
        "runtime_seconds": result.runtime_seconds,
        "error_message": result.error_message,
    }


def build_batch_leaderboard(results: list[ExperimentResult]) -> pd.DataFrame:
    rows = [_result_to_row(result) for result in results]
    if not rows:
        return pd.DataFrame(
            columns=[
                "experiment_id",
                "strategy_family",
                "config_hash",
                "strategy_type",
                "source_type",
                "template_id",
                "hypothesis",
                "reason_selected",
                "novelty_score",
                "selection_score",
                "exploration_mode",
                "proposal_role",
                "region_label",
                "duplicate_risk",
                "dead_zone_risk",
                "parent_config_hash",
                "near_duplicate_of",
                "dead_zone_flags",
                "source_proposal_id",
                "source_region_id",
                "source_grid_search_id",
                "source_cells",
                "region_class",
                "promotion_recommendation",
                "region_state",
                "neighbor_of_cell",
                "neighbor_generation_reason",
                "neighbor_grid_status",
                "validator_mode",
                "allowed_override_params",
                "allowed_override_values",
                "off_grid_params",
                "validation_override_reason",
                "source_idea_ids",
                "confirmation_state",
                "confirmation_required",
                "confirmation_reason",
                "confirmation_batch_id",
                "confirmation_trial_kind",
                "targeted_follow_up_required",
                "targeted_follow_up_reason",
                "targeted_follow_up_type",
                "targeted_follow_up_priority",
                "targeted_follow_up_batch_id",
                "holdout_check_required",
                "holdout_check_type",
                "holdout_check_status",
                "holdout_check_outcome",
                "holdout_check_scope",
                "holdout_check_batch_id",
                "holdout_horizon_tags",
                "holdout_regime_tags",
                "objective_score",
                "sharpe",
                "calmar",
                "total_return",
                "max_drawdown",
                "trades_per_year",
                "exposure",
                "status",
                "viable",
                "baseline_name",
                "comparison_status",
                "delta_sharpe",
                "delta_calmar",
                "delta_return",
                "beats_baseline_objective",
                "beats_baseline_guardrails",
                "runtime_seconds",
                "error_message",
            ]
        )

    eligible = [
        {
            "objective_score": result.objective_score,
            "metrics": result.metrics,
            "robustness": result.robustness,
            "row": row,
        }
        for result, row in zip(results, rows, strict=False)
        if result.status not in {"duplicate", "invalid", "error", "skipped"}
    ]
    ranked_rows = [entry["row"] for entry in rank_results(eligible)] if eligible else []
    in_ranked = {row["experiment_id"] for row in ranked_rows}
    trailing_rows = [row for row in rows if row["experiment_id"] not in in_ranked]
    return pd.DataFrame(ranked_rows + trailing_rows)


_PROPOSAL_METADATA_BULK_KEYS = frozenset({
    # Large branch-planning structures written by build_branch_budget_plan;
    # hundreds of KB per batch but not needed once the batch has run.
    "families",
    "branch_budgets",
    "branch_budget_rationale",
    "history",  # any residual history blob
})
_RUNTIME_DECISION_BULK_KEYS = frozenset({
    "rationale",
    "branch_budgets",
    "used_signals",
})


def _strip_proposal_metadata_for_disk(metadata: Any) -> Any:
    """Return a copy of proposal_metadata with bulk branch-plan fields removed.

    Reduces per-batch summary.json from ~880 KB to ~30-50 KB without losing
    any operationally-needed fields (confirmation state, quality flags, etc.).
    """
    if not isinstance(metadata, dict):
        return metadata
    stripped = {k: v for k, v in metadata.items() if k not in _PROPOSAL_METADATA_BULK_KEYS}
    if isinstance(stripped.get("runtime_decision"), dict):
        stripped["runtime_decision"] = {
            k: v
            for k, v in stripped["runtime_decision"].items()
            if k not in _RUNTIME_DECISION_BULK_KEYS
        }
    return stripped


def build_batch_summary(batch_result: BatchResult) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    family_summary: dict[str, dict[str, Any]] = {}
    proposal_metadata = batch_result.request.proposal_metadata or {}
    confirmation_meta = proposal_metadata.get("confirmation") if isinstance(proposal_metadata.get("confirmation"), dict) else {}
    proposal_quality = proposal_metadata.get("proposal_quality") if isinstance(proposal_metadata.get("proposal_quality"), dict) else {}
    throughput_diagnostics = proposal_metadata.get("throughput_diagnostics") if isinstance(proposal_metadata.get("throughput_diagnostics"), dict) else {}
    if throughput_diagnostics:
        throughput_diagnostics = {**throughput_diagnostics, "process_memory_kb": current_process_memory_kb()}

    for result in batch_result.results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        family = result.spec.family
        family_bucket = family_summary.setdefault(
            family,
            {
                "results": 0,
                "executed": 0,
                "skipped": 0,
                "failed": 0,
                "best_objective_score": None,
                "best_experiment_id": None,
                "strategy_type_counts": {},
                "source_type_counts": {},
                "template_counts": {},
            },
        )
        family_bucket["results"] += 1
        if result.status in {"success", "no_trades", "duplicate"}:
            family_bucket["executed"] += 1
        if result.status in {"duplicate", "skipped"}:
            family_bucket["skipped"] += 1
        if result.status in {"error", "invalid"}:
            family_bucket["failed"] += 1
        if result.objective_score is not None and (
            family_bucket["best_objective_score"] is None
            or float(result.objective_score) > float(family_bucket["best_objective_score"])
        ):
            family_bucket["best_objective_score"] = result.objective_score
            family_bucket["best_experiment_id"] = result.spec.experiment_id
        strategy_type = result.spec.strategy_type or "unspecified"
        family_bucket["strategy_type_counts"][strategy_type] = family_bucket["strategy_type_counts"].get(strategy_type, 0) + 1
        source_type = result.spec.source_type or "unspecified"
        family_bucket["source_type_counts"][source_type] = family_bucket["source_type_counts"].get(source_type, 0) + 1
        template_id = result.spec.template_id or "unspecified"
        family_bucket["template_counts"][template_id] = family_bucket["template_counts"].get(template_id, 0) + 1

    return {
        "batch_id": batch_result.request.batch_id,
        "timestamp_utc": batch_result.request.timestamp_utc,
        "strategy_families": batch_result.request.strategy_families,
        "sampler_type": batch_result.request.sampler_type,
        "max_workers": getattr(batch_result, "max_workers", None),
        "execution_mode": getattr(batch_result, "execution_mode", "sequential"),
        "worker_failures": getattr(batch_result, "worker_failures", 0),
        "source_proposal_id": batch_result.request.source_proposal_id,
        "proposal_metadata": _strip_proposal_metadata_for_disk(batch_result.request.proposal_metadata),
        "proposal_quality": proposal_quality,
        "throughput_diagnostics": throughput_diagnostics,
        "total_sampled": batch_result.total_sampled,
        "total_executed": batch_result.total_executed,
        "total_skipped": batch_result.total_skipped,
        "total_failed": batch_result.total_failed,
        "status_counts": status_counts,
        "family_summary": family_summary,
        "confirmation_state": proposal_metadata.get("confirmation_state") or confirmation_meta.get("state"),
        "confirmation_required": proposal_metadata.get("confirmation_required") if proposal_metadata.get("confirmation_required") is not None else confirmation_meta.get("required"),
        "confirmation_reason": proposal_metadata.get("confirmation_reason") or confirmation_meta.get("reason"),
        "confirmation_batch_id": proposal_metadata.get("confirmation_batch_id") or confirmation_meta.get("batch_id"),
        "confirmation_outcome": proposal_metadata.get("confirmation_outcome") or confirmation_meta.get("outcome"),
        "promotion_state": proposal_metadata.get("promotion_state") or confirmation_meta.get("state"),
        "targeted_follow_up_required": proposal_metadata.get("targeted_follow_up_required"),
        "targeted_follow_up_reason": proposal_metadata.get("targeted_follow_up_reason"),
        "targeted_follow_up_type": proposal_metadata.get("targeted_follow_up_type"),
        "targeted_follow_up_priority": proposal_metadata.get("targeted_follow_up_priority"),
        "targeted_follow_up_batch_id": proposal_metadata.get("targeted_follow_up_batch_id"),
        "holdout_check_required": proposal_metadata.get("holdout_check_required"),
        "holdout_check_type": proposal_metadata.get("holdout_check_type"),
        "holdout_check_status": proposal_metadata.get("holdout_check_status"),
        "holdout_check_outcome": proposal_metadata.get("holdout_check_outcome"),
        "holdout_check_scope": proposal_metadata.get("holdout_check_scope"),
        "holdout_check_batch_id": proposal_metadata.get("holdout_check_batch_id"),
        "holdout_horizon_tags": proposal_metadata.get("holdout_horizon_tags"),
        "holdout_regime_tags": proposal_metadata.get("holdout_regime_tags"),
    }


def save_batch_reports(batch_result: BatchResult, *, base_dir: str) -> dict[str, str]:
    batch_dir = Path(base_dir) / "batches" / batch_result.request.batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = build_batch_leaderboard(batch_result.results)
    raw_results = pd.DataFrame([_result_to_row(result) for result in batch_result.results])
    summary = build_batch_summary(batch_result)

    leaderboard_path = batch_dir / "leaderboard.csv"
    raw_results_path = batch_dir / "raw_results.csv"
    summary_path = batch_dir / "summary.json"

    leaderboard.to_csv(leaderboard_path, index=False)
    raw_results.to_csv(raw_results_path, index=False)
    summary["leaderboard_path"] = str(leaderboard_path)
    summary["raw_results_path"] = str(raw_results_path)
    summary["summary_path"] = str(summary_path)
    if not leaderboard.empty:
        top_results = leaderboard.copy()
        if "objective_score" in top_results.columns:
            top_results["_objective_score"] = pd.to_numeric(top_results["objective_score"], errors="coerce")
            top_results = top_results.sort_values("_objective_score", ascending=False, na_position="last")
        summary["top_results"] = top_results.head(5).drop(columns=["_objective_score"], errors="ignore").to_dict("records")
        if "viable" in leaderboard.columns:
            summary["viable_count"] = int(leaderboard["viable"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        if "beats_baseline_objective" in leaderboard.columns:
            summary["baseline_beating_count"] = int(
                leaderboard["beats_baseline_objective"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
            )
    try:
        upsert_batch_summary(base_dir, summary)
        refresh_hot_index_reports(
            base_dir=base_dir,
            families=list(batch_result.request.strategy_families),
            recent_limit_per_family=max(32, min(128, int(batch_result.request.max_experiments or 32))),
        )
    except Exception:
        pass
    disk_limit = int(os.environ.get("BATCH_SUMMARY_DISK_LIMIT_BYTES", str(BATCH_SUMMARY_DISK_LIMIT_BYTES)))
    disk_summary = compact_batch_summary_for_disk(summary, limit_bytes=disk_limit)
    disk_summary["artifact_compaction"] = {
        **(disk_summary.get("artifact_compaction") if isinstance(disk_summary.get("artifact_compaction"), dict) else {}),
        "mode": "write_compact_summary",
        "disk_limit_bytes": disk_limit,
        "sqlite_written": True,
    }
    summary_path.write_text(json.dumps(disk_summary, indent=2, sort_keys=True, default=str))

    return {
        "leaderboard_path": str(leaderboard_path),
        "raw_results_path": str(raw_results_path),
        "summary_path": str(summary_path),
    }


def run_batch_experiments(
    request: BatchRequest,
    *,
    data: dict[str, Any] | None = None,
    base_dir: str = "experiments",
    baseline_by_family: dict[str, str | None] | None = None,
) -> BatchResult:
    init_store(base_dir)
    shared_data = data or load_data()
    baseline_by_family = {**DEFAULT_BASELINES, **(baseline_by_family or {})}
    batch_started = time.perf_counter()

    requested_total = 0
    skipped = 0
    failed = 0
    execution_specs: list[ExperimentSpec] = []
    already_seen_in_batch: set[tuple[str, str]] = set()
    worker_count = request.max_workers if request.max_workers is not None else (1 if request.execution_mode == "sequential" else 6)
    if worker_count > 8:
        raise ValueError("max_workers must be <= 8.")
    execution_mode = "parallel" if worker_count > 1 else "sequential"

    for family_idx, family in enumerate(request.strategy_families):
        if request.precomputed_specs and family in request.precomputed_specs:
            specs = list(request.precomputed_specs[family])
        else:
            specs = []
            if request.precomputed_configs and family in request.precomputed_configs:
                configs = [dict(config) for config in request.precomputed_configs[family]]
            else:
                configs = sample_batch_configs(
                    family=family,
                    method=request.sampler_type,
                    n=request.max_per_family,
                    seed=request.seed + family_idx,
                )
            for config in configs:
                config_hash = compute_config_hash(family, config)
                specs.append(
                    ExperimentSpec(
                        family=family,
                        params=config,
                        search_method=request.sampler_type,
                        objective_name=request.objective_name,
                        batch_id=request.batch_id,
                        config_hash=config_hash,
                        experiment_id=f"{family}_{config_hash}_{request.batch_id}",
                        timestamp_utc=request.timestamp_utc,
                        strategy_type=DEFAULT_STRATEGY_TYPES.get(family),
                    )
                )
        prior = load_prior_results(family=family, base_dir=base_dir) if request.resume else pd.DataFrame()
        successful_prior_hashes = (
            set(prior.loc[prior["status"] == "success", "config_hash"].astype(str)) if not prior.empty else set()
        )

        for spec in specs:
            config = dict(spec.params)
            if requested_total >= request.max_experiments:
                break
            requested_total += 1
            config_hash = spec.config_hash or compute_config_hash(family, config)
            batch_key = (family, config_hash)
            if batch_key in already_seen_in_batch:
                skipped += 1
                continue
            already_seen_in_batch.add(batch_key)

            if request.resume and config_hash in successful_prior_hashes and not _allows_confirmation_reproduction(spec):
                skipped += 1
                continue

            execution_specs.append(spec)
        if requested_total >= request.max_experiments:
            break

    if execution_specs:
        if worker_count > 1:
            computed_results, worker_failures = run_experiments_parallel(
                execution_specs,
                data=shared_data,
                base_dir=base_dir,
                max_workers=worker_count,
                fail_fast=request.fail_fast,
                baseline_by_family=baseline_by_family,
            )
        else:
            computed_results = [
                run_single_experiment(
                    family=spec.family,
                    config=dict(spec.params),
                    data=shared_data,
                    persist=False,
                    base_dir=base_dir,
                    compare_to_baseline=baseline_by_family.get(spec.family),
                    experiment_id=spec.experiment_id or None,
                    spec=spec,
                )
                for spec in execution_specs
            ]
            worker_failures = 0
    else:
        computed_results = []
        worker_failures = 0
    shared_data = None

    results: list[ExperimentResult] = []
    if request.persist:
        for result in computed_results:
            payload = asdict(result)
            payload["spec"] = asdict(result.spec)
            persisted = save_experiment_result_atomic(payload, base_dir=base_dir)
            if not persisted and result.status in {"success", "no_trades"}:
                result = replace(result, status="duplicate")
            results.append(result)
    else:
        results = list(computed_results)

    executed = 0
    for result in results:
        if result.status == "duplicate":
            skipped += 1
        elif result.status in {"error", "invalid"}:
            failed += 1
        elif result.status in {"success", "no_trades"}:
            executed += 1

    batch_result = BatchResult(
        request=request,
        status="completed",
        total_sampled=requested_total,
        total_executed=executed,
        total_skipped=skipped,
        total_failed=failed,
        results=results,
        max_workers=worker_count,
        execution_mode=execution_mode,
        worker_failures=worker_failures,
    )
    report_paths = save_batch_reports(batch_result, base_dir=base_dir)
    batch_runtime_seconds = round(time.perf_counter() - batch_started, 3)
    total_runtime_seconds = sum(float(result.runtime_seconds or 0.0) for result in batch_result.results)
    proposal_metadata = request.proposal_metadata or {}
    throughput_diagnostics = {
        "batch_runtime_seconds": batch_runtime_seconds,
        "executed_per_minute": round((batch_result.total_executed / max(batch_runtime_seconds, 1e-9)) * 60.0, 3),
        "worker_utilization_estimate": round(
            total_runtime_seconds / max(batch_runtime_seconds * max(worker_count, 1), 1e-9),
            3,
        ),
        "requested_candidates": requested_total,
        "sampled_candidates": batch_result.total_sampled,
        "executed_candidates": batch_result.total_executed,
        "skipped_candidates": batch_result.total_skipped,
        "failed_candidates": batch_result.total_failed,
        "worker_count": worker_count,
        "execution_mode": execution_mode,
        "process_memory_kb": current_process_memory_kb(),
        "filter_reasons": (proposal_metadata.get("proposal_quality") or {}).get("shortfall_reasons")
        if isinstance(proposal_metadata, dict)
        else None,
    }
    summary_path = Path(report_paths["summary_path"])
    summary_payload = json.loads(summary_path.read_text())
    summary_payload["throughput_diagnostics"] = throughput_diagnostics
    summary_payload["batch_runtime_seconds"] = batch_runtime_seconds
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    return BatchResult(
        request=batch_result.request,
        status=batch_result.status,
        total_sampled=batch_result.total_sampled,
        total_executed=batch_result.total_executed,
        total_skipped=batch_result.total_skipped,
        total_failed=batch_result.total_failed,
        results=batch_result.results,
        max_workers=batch_result.max_workers,
        execution_mode=batch_result.execution_mode,
        worker_failures=batch_result.worker_failures,
        leaderboard_path=report_paths["leaderboard_path"],
        raw_results_path=report_paths["raw_results_path"],
        summary_path=report_paths["summary_path"],
    )


def proposal_to_batch_request(
    proposal: ProposalResult,
    *,
    persist: bool = True,
    resume: bool = True,
    max_workers: int | None = None,
    fail_fast: bool = False,
) -> BatchRequest:
    max_per_family = max((len(configs) for configs in proposal.candidate_configs.values()), default=0)
    precomputed_specs: dict[str, list[ExperimentSpec]] = {}
    for family, configs in proposal.candidate_configs.items():
        metadata_items = (proposal.candidate_metadata or {}).get(family, [])
        specs: list[ExperimentSpec] = []
        for idx, config in enumerate(configs):
            metadata = metadata_items[idx] if idx < len(metadata_items) else {}
            config_hash = compute_config_hash(family, config)
            specs.append(
                ExperimentSpec(
                    family=family,
                    params=config,
                    search_method="proposal",
                    objective_name=proposal.request.objective_name,
                    batch_id=f"{proposal.request.proposal_id}_batch",
                    config_hash=config_hash,
                    experiment_id=f"{family}_{config_hash}_{proposal.request.proposal_id}",
                    timestamp_utc=proposal.request.timestamp_utc,
                    strategy_type=metadata.get("strategy_type") or DEFAULT_STRATEGY_TYPES.get(family),
                    source_type=metadata.get("source_type"),
                    template_id=metadata.get("template_id"),
                    hypothesis=metadata.get("hypothesis"),
                    reason_selected=metadata.get("reason_selected"),
                    novelty_score=metadata.get("novelty_score"),
                    exploration_mode=metadata.get("exploration_mode"),
                    proposal_role=metadata.get("proposal_role"),
                    region_label=metadata.get("region_label"),
                    duplicate_risk=metadata.get("duplicate_risk"),
                    dead_zone_risk=metadata.get("dead_zone_risk"),
                    parent_config_hash=metadata.get("parent_config_hash"),
                    near_duplicate_of=metadata.get("near_duplicate_of"),
                    dead_zone_flags=metadata.get("dead_zone_flags"),
                    selection_score=metadata.get("selection_score"),
                    source_proposal_id=proposal.request.proposal_id,
                    source_region_id=metadata.get("source_region_id"),
                    source_grid_search_id=metadata.get("source_grid_search_id"),
                    source_cells=metadata.get("source_cells"),
                    region_class=metadata.get("region_class"),
                    promotion_recommendation=metadata.get("promotion_recommendation"),
                    region_state=metadata.get("region_state"),
                    neighbor_of_cell=metadata.get("neighbor_of_cell"),
                    neighbor_generation_reason=metadata.get("neighbor_generation_reason"),
                    neighbor_grid_status=metadata.get("neighbor_grid_status"),
                    validator_mode=metadata.get("validator_mode"),
                    allowed_override_params=metadata.get("allowed_override_params"),
                    allowed_override_values=metadata.get("allowed_override_values"),
                    off_grid_params=metadata.get("off_grid_params"),
                    validation_override_reason=metadata.get("validation_override_reason"),
                    source_idea_ids=metadata.get("source_idea_ids"),
                    idea_id=metadata.get("idea_id"),
                    idea_source=metadata.get("idea_source"),
                    idea_kind=metadata.get("idea_kind"),
                    novelty_reason=metadata.get("novelty_reason"),
                    is_new_idea=bool(metadata.get("is_new_idea")) if metadata.get("is_new_idea") is not None else False,
                    is_structurally_novel=bool(metadata.get("is_structurally_novel"))
                    if metadata.get("is_structurally_novel") is not None
                    else False,
                    is_uncommon_idea=bool(metadata.get("is_uncommon_idea"))
                    if metadata.get("is_uncommon_idea") is not None
                    else False,
                    is_branch_repeat=bool(metadata.get("is_branch_repeat")) if metadata.get("is_branch_repeat") is not None else False,
                    repeat_branch_flag=bool(metadata.get("repeat_branch_flag")) if metadata.get("repeat_branch_flag") is not None else False,
                    repeat_branch_depth=metadata.get("repeat_branch_depth"),
                    new_idea_budget_bucket=metadata.get("new_idea_budget_bucket"),
                    uncommon_idea_reason=metadata.get("uncommon_idea_reason"),
                    synthesized_template_family=metadata.get("synthesized_template_family"),
                    synthesis_rationale=metadata.get("synthesis_rationale"),
                    confirmation_state=metadata.get("confirmation_state") or proposal.request.confirmation_state,
                    confirmation_required=bool(
                        metadata.get("confirmation_required")
                        if metadata.get("confirmation_required") is not None
                        else proposal.request.confirmation_required
                    ),
                    confirmation_reason=metadata.get("confirmation_reason") or proposal.request.confirmation_reason,
                    confirmation_batch_id=metadata.get("confirmation_batch_id") or proposal.request.confirmation_batch_id,
                    confirmation_trial_kind=metadata.get("confirmation_trial_kind"),
                    targeted_follow_up_required=bool(
                        metadata.get("targeted_follow_up_required")
                        if metadata.get("targeted_follow_up_required") is not None
                        else proposal.request.targeted_follow_up_required
                    ),
                    targeted_follow_up_reason=metadata.get("targeted_follow_up_reason") or proposal.request.targeted_follow_up_reason,
                    targeted_follow_up_type=metadata.get("targeted_follow_up_type") or proposal.request.targeted_follow_up_type,
                    targeted_follow_up_priority=metadata.get("targeted_follow_up_priority")
                    if metadata.get("targeted_follow_up_priority") is not None
                    else proposal.request.targeted_follow_up_priority,
                    targeted_follow_up_batch_id=metadata.get("targeted_follow_up_batch_id")
                    or proposal.request.targeted_follow_up_batch_id,
                    holdout_check_required=bool(
                        metadata.get("holdout_check_required")
                        if metadata.get("holdout_check_required") is not None
                        else proposal.request.holdout_check_required
                    ),
                    holdout_check_type=metadata.get("holdout_check_type") or proposal.request.holdout_check_type,
                    holdout_check_status=metadata.get("holdout_check_status") or proposal.request.holdout_check_status,
                    holdout_check_outcome=metadata.get("holdout_check_outcome") or proposal.request.holdout_check_outcome,
                    holdout_check_scope=metadata.get("holdout_check_scope") or proposal.request.holdout_check_scope,
                    holdout_check_batch_id=metadata.get("holdout_check_batch_id") or proposal.request.holdout_check_batch_id,
                    holdout_horizon_tags=metadata.get("holdout_horizon_tags") or proposal.request.holdout_horizon_tags,
                    holdout_regime_tags=metadata.get("holdout_regime_tags") or proposal.request.holdout_regime_tags,
                )
            )
        precomputed_specs[family] = specs
    return BatchRequest(
        batch_id=f"{proposal.request.proposal_id}_batch",
        timestamp_utc=proposal.request.timestamp_utc,
        strategy_families=proposal.request.strategy_families,
        sampler_type="random",
        max_experiments=proposal.request.max_experiments,
        max_per_family=max_per_family,
        seed=proposal.request.seed,
        persist=persist,
        resume=resume and proposal.request.resume,
        objective_name=proposal.request.objective_name,
        max_workers=max_workers,
        fail_fast=fail_fast,
        include_filters=None,
        exclude_filters=None,
        precomputed_configs=proposal.candidate_configs,
        precomputed_specs=precomputed_specs,
        source_proposal_id=proposal.request.proposal_id,
        proposal_metadata=proposal.reasoning_summary,
    )


def _allows_confirmation_reproduction(spec: ExperimentSpec) -> bool:
    if not bool(getattr(spec, "confirmation_required", False)):
        return False
    if str(getattr(spec, "confirmation_trial_kind", "") or "") not in {"reproduce", "confirmation_reproduce"}:
        return False
    return bool(getattr(spec, "confirmation_batch_id", None))
