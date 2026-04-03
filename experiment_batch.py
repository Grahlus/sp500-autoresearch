from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from experiment_objective import rank_results
from experiment_runner import run_single_experiment
from experiment_spaces import enumerate_grid_candidates, list_searchable_families, sample_random_candidates
from experiment_store import compute_config_hash, init_store, load_prior_results
from experiment_types import BatchRequest, BatchResult, ExperimentResult, ProposalResult
from prepare import load_data


DEFAULT_BASELINES = {
    "momentum": "momentum_champion_s10005",
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
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        precomputed_configs=None,
        source_proposal_id=None,
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
    return {
        "experiment_id": result.spec.experiment_id,
        "strategy_family": result.spec.family,
        "config_hash": result.spec.config_hash,
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


def build_batch_summary(batch_result: BatchResult) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    family_summary: dict[str, dict[str, Any]] = {}

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

    return {
        "batch_id": batch_result.request.batch_id,
        "timestamp_utc": batch_result.request.timestamp_utc,
        "strategy_families": batch_result.request.strategy_families,
        "sampler_type": batch_result.request.sampler_type,
        "source_proposal_id": batch_result.request.source_proposal_id,
        "total_sampled": batch_result.total_sampled,
        "total_executed": batch_result.total_executed,
        "total_skipped": batch_result.total_skipped,
        "total_failed": batch_result.total_failed,
        "status_counts": status_counts,
        "family_summary": family_summary,
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

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

    requested_total = 0
    executed = 0
    skipped = 0
    failed = 0
    results: list[ExperimentResult] = []
    already_seen_in_batch: set[tuple[str, str]] = set()

    for family_idx, family in enumerate(request.strategy_families):
        if request.precomputed_configs and family in request.precomputed_configs:
            configs = [dict(config) for config in request.precomputed_configs[family]]
        else:
            configs = sample_batch_configs(
                family=family,
                method=request.sampler_type,
                n=request.max_per_family,
                seed=request.seed + family_idx,
            )
        prior = load_prior_results(family=family, base_dir=base_dir) if request.resume else pd.DataFrame()
        successful_prior_hashes = (
            set(prior.loc[prior["status"] == "success", "config_hash"].astype(str)) if not prior.empty else set()
        )

        for config in configs:
            if requested_total >= request.max_experiments:
                break
            requested_total += 1

            config_hash = compute_config_hash(family, config)
            batch_key = (family, config_hash)
            if batch_key in already_seen_in_batch:
                skipped += 1
                continue
            already_seen_in_batch.add(batch_key)

            if request.resume and config_hash in successful_prior_hashes:
                skipped += 1
                continue

            result = run_single_experiment(
                family=family,
                config=config,
                data=shared_data,
                persist=request.persist,
                base_dir=base_dir,
                compare_to_baseline=baseline_by_family.get(family),
            )
            results.append(result)
            if result.status in {"duplicate"}:
                skipped += 1
            elif result.status in {"error", "invalid"}:
                failed += 1
            else:
                executed += 1
        if requested_total >= request.max_experiments:
            break

    batch_result = BatchResult(
        request=request,
        status="completed",
        total_sampled=requested_total,
        total_executed=executed,
        total_skipped=skipped,
        total_failed=failed,
        results=results,
    )
    report_paths = save_batch_reports(batch_result, base_dir=base_dir)
    return BatchResult(
        request=batch_result.request,
        status=batch_result.status,
        total_sampled=batch_result.total_sampled,
        total_executed=batch_result.total_executed,
        total_skipped=batch_result.total_skipped,
        total_failed=batch_result.total_failed,
        results=batch_result.results,
        leaderboard_path=report_paths["leaderboard_path"],
        raw_results_path=report_paths["raw_results_path"],
        summary_path=report_paths["summary_path"],
    )


def proposal_to_batch_request(
    proposal: ProposalResult,
    *,
    persist: bool = True,
    resume: bool = True,
) -> BatchRequest:
    max_per_family = max((len(configs) for configs in proposal.candidate_configs.values()), default=0)
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
        include_filters=None,
        exclude_filters=None,
        precomputed_configs=proposal.candidate_configs,
        source_proposal_id=proposal.request.proposal_id,
    )
