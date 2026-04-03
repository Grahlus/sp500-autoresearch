from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from baseline_registry import get_baseline
from prepare import load_data, run_backtest
from strategies import get_strategy_family

from experiment_objective import evaluate_objective, evaluate_robustness
from experiment_spaces import normalize_experiment_config, validate_experiment_config
from experiment_store import compute_config_hash, has_experiment_result, load_experiment_result, save_experiment_result
from experiment_types import ExperimentResult, ExperimentSpec


def _resolve_benchmark_source(data: dict[str, Any]) -> str:
    close = data.get("close")
    if isinstance(close, pd.DataFrame) and "SPY" in close.columns:
        return "spy_symbol"
    if isinstance(data.get("spy_close"), pd.Series):
        return "spy_close_column"
    if isinstance(data.get("benchmark_close"), pd.Series):
        return "benchmark_close_column"
    return "equal_weight_proxy"


def _derive_dataset_identity(data: dict[str, Any]) -> tuple[str, str, str]:
    close = data["close"]
    data_start = str(close.index[0].date())
    data_end = str(close.index[-1].date())
    payload = json.dumps(
        {
            "start": data_start,
            "end": data_end,
            "rows": int(close.shape[0]),
            "cols": int(close.shape[1]),
            "columns": close.columns.tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return dataset_id, data_start, data_end


def _resolve_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
            or None
        )
    except Exception:
        return None


def _derive_extended_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    enriched["total_return"] = metrics.get("total_return_pct")
    final_values = [window.get("final_value") for window in metrics.get("windows", []) if window.get("final_value") is not None]
    avg_window_days = [window.get("n_days") for window in metrics.get("windows", []) if window.get("n_days") is not None]

    if final_values:
        enriched["final_value"] = round(float(sum(final_values) / len(final_values)), 2)
    else:
        enriched["final_value"] = None

    if avg_window_days and metrics.get("total_return_pct") is not None:
        avg_days = float(sum(avg_window_days) / len(avg_window_days))
        total_return = float(metrics["total_return_pct"]) / 100.0
        enriched["annual_return"] = round((((1.0 + total_return) ** (252.0 / max(avg_days, 1.0))) - 1.0) * 100.0, 4)
    else:
        enriched["annual_return"] = None

    enriched["turnover"] = None
    enriched["exposure"] = None
    enriched["trade_count"] = None
    enriched["avg_hold_days"] = None
    return enriched


def _build_baseline_comparison(
    spec: ExperimentSpec,
    metrics: dict[str, Any],
    objective_score: float | None,
    robustness: dict[str, Any],
    baseline_name: str | None,
) -> dict[str, Any] | None:
    if not baseline_name:
        return None

    baseline = get_baseline(baseline_name)
    if baseline["family"] != spec.family:
        raise ValueError(
            f"Baseline '{baseline_name}' belongs to family '{baseline['family']}', not '{spec.family}'."
        )

    comparison_policy = baseline.get("comparison_policy", {})
    comparison: dict[str, Any] = {
        "baseline_name": baseline_name,
        "baseline_config_hash": baseline.get("config_hash"),
        "baseline_metric_source": comparison_policy.get("metrics_verification_status", "historical_note_only"),
        "comparison_eligible": bool(comparison_policy.get("comparison_eligible", False)),
        "comparison_kind": "partial",
        "comparison_status": "historical_provenance_only",
        "baseline_verified": False,
        "baseline_source_of_truth": baseline.get("source_of_truth"),
        "baseline_config_source": baseline.get("config_source", baseline.get("baseline_config_source")),
        "baseline_exact_config_match": baseline.get("config_hash") == spec.config_hash,
        "comparison_notes": comparison_policy.get("comparison_rule"),
        "delta_sharpe": None,
        "delta_calmar": None,
        "delta_return": None,
        "beats_baseline_objective": None,
        "beats_baseline_guardrails": None,
    }

    verified_run = baseline.get("verified_run")
    verified_metrics = baseline.get("verified_metrics")
    if verified_run and verified_metrics:
        comparison["baseline_verified"] = True
        exact = (
            verified_run.get("dataset_id") == spec.dataset_id
            and verified_run.get("split") == spec.split
            and verified_run.get("benchmark_source") == spec.benchmark_source
        )
        comparison["comparison_kind"] = "exact" if exact else "partial"
        comparison["comparison_status"] = "exact_verified_current_engine" if exact else "partial_verified_current_engine"
        comparison["baseline_dataset_id"] = verified_run.get("dataset_id")
        comparison["baseline_split"] = verified_run.get("split")
        comparison["baseline_benchmark_source"] = verified_run.get("benchmark_source")
        comparison["baseline_sharpe"] = verified_metrics.get("sharpe")
        comparison["baseline_calmar"] = verified_metrics.get("calmar")
        comparison["baseline_total_return"] = verified_metrics.get("total_return")
        comparison["baseline_objective_score"] = verified_run.get("objective_score")
        comparison["baseline_robustness"] = verified_run.get("robustness")
        if metrics.get("sharpe") is not None and verified_metrics.get("sharpe") is not None:
            delta = round(float(metrics["sharpe"]) - float(verified_metrics["sharpe"]), 6)
            comparison["sharpe_delta"] = delta
            comparison["delta_sharpe"] = delta
        if metrics.get("calmar") is not None and verified_metrics.get("calmar") is not None:
            comparison["delta_calmar"] = round(float(metrics["calmar"]) - float(verified_metrics["calmar"]), 6)
        if metrics.get("total_return") is not None and verified_metrics.get("total_return") is not None:
            delta = round(
                float(metrics["total_return"]) - float(verified_metrics["total_return"]), 6
            )
            comparison["total_return_delta"] = delta
            comparison["delta_return"] = delta
        if objective_score is not None and verified_run.get("objective_score") is not None:
            comparison["beats_baseline_objective"] = bool(objective_score > float(verified_run["objective_score"]))
        if verified_run.get("robustness") is not None:
            current_viable = bool(robustness.get("viable"))
            baseline_viable = bool(verified_run["robustness"].get("viable"))
            comparison["beats_baseline_guardrails"] = bool(int(current_viable) >= int(baseline_viable))
    else:
        note = baseline.get("historical_provenance", {}).get("starting_point_champion_note", {})
        comparison["baseline_historical_note"] = {
            "reported_val": note.get("reported_val"),
            "reported_train": note.get("reported_train"),
            "reported_oos": note.get("reported_oos"),
            "reported_config_note": note.get("reported_config_note"),
            "verification_status": note.get("verification_status"),
        }

    return comparison


def _make_experiment_spec(
    family: str,
    normalized_config: dict[str, Any],
    data: dict[str, Any],
    experiment_id: str | None = None,
) -> ExperimentSpec:
    strategy = get_strategy_family(family)
    dataset_id, data_start, data_end = _derive_dataset_identity(data)
    timestamp_utc = datetime.now(UTC).isoformat()
    config_hash = compute_config_hash(family, normalized_config)
    experiment_id = experiment_id or f"{family}_{config_hash}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    return ExperimentSpec(
        family=family,
        params=normalized_config,
        search_method="single",
        objective_name="wf_v1_score",
        batch_id="single",
        config_hash=config_hash,
        experiment_id=experiment_id,
        timestamp_utc=timestamp_utc,
        benchmark_source=_resolve_benchmark_source(data),
        dataset_id=dataset_id,
        data_start=data_start,
        data_end=data_end,
        split="walk-forward",
        git_commit=_resolve_git_commit(),
        family_version=strategy.hypothesis,
    )


def _make_invalid_spec(
    family: str,
    config: dict[str, Any],
    data: dict[str, Any],
    experiment_id: str | None = None,
) -> ExperimentSpec:
    dataset_id, data_start, data_end = _derive_dataset_identity(data)
    timestamp_utc = datetime.now(UTC).isoformat()
    normalized_family = family.strip().lower()
    return ExperimentSpec(
        family=normalized_family,
        params=dict(config or {}),
        search_method="single",
        objective_name="wf_v1_score",
        batch_id="single",
        config_hash=compute_config_hash(normalized_family, dict(sorted((config or {}).items()))),
        experiment_id=experiment_id or f"{normalized_family}_invalid_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        timestamp_utc=timestamp_utc,
        benchmark_source=_resolve_benchmark_source(data),
        dataset_id=dataset_id,
        data_start=data_start,
        data_end=data_end,
        split="walk-forward",
        git_commit=_resolve_git_commit(),
        family_version=None,
    )


def build_strategy_weights_for_experiment(
    family: str,
    data: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    strategy = get_strategy_family(family)
    if strategy.generate_signals_with_config is not None:
        return strategy.generate_signals_with_config(data, config)
    return strategy.generate_signals(data)


def _result_from_loaded(payload: dict[str, Any], status: str) -> ExperimentResult:
    spec_dict = payload["spec"]
    result_spec = ExperimentSpec(
        family=spec_dict["family"],
        params=spec_dict["params"],
        search_method=spec_dict.get("search_method", "single"),
        objective_name=spec_dict.get("objective_name", "wf_v1_score"),
        batch_id=spec_dict.get("batch_id", "single"),
        config_hash=spec_dict.get("config_hash", ""),
        experiment_id=spec_dict.get("experiment_id", ""),
        timestamp_utc=spec_dict.get("timestamp_utc", ""),
        benchmark_source=spec_dict.get("benchmark_source"),
        dataset_id=spec_dict.get("dataset_id"),
        data_start=spec_dict.get("data_start"),
        data_end=spec_dict.get("data_end"),
        split=spec_dict.get("split", "walk-forward"),
        git_commit=spec_dict.get("git_commit"),
        family_version=spec_dict.get("family_version"),
    )
    return ExperimentResult(
        spec=result_spec,
        status=status,
        objective_score=payload.get("objective_score", payload.get("score")),
        metrics=payload.get("metrics", {}),
        robustness=payload.get("robustness", {}),
        artifacts=payload.get("artifacts", {}),
        baseline_comparison=payload.get("baseline_comparison"),
        error_message=payload.get("error_message", payload.get("error")),
        runtime_seconds=payload.get("runtime_seconds"),
    )


def run_single_experiment(
    family: str,
    config: dict[str, Any],
    *,
    data: dict[str, Any] | None = None,
    persist: bool = True,
    base_dir: str = "experiments",
    experiment_id: str | None = None,
    compare_to_baseline: str | None = None,
) -> ExperimentResult:
    started = time.time()
    data = data or load_data()

    try:
        normalized_config = normalize_experiment_config(family, config)
    except ValueError as exc:
        spec = _make_invalid_spec(family, dict(config or {}), data, experiment_id=experiment_id)
        return ExperimentResult(
            spec=spec,
            status="invalid",
            objective_score=None,
            metrics={},
            robustness={"viable": False, "negative_windows": 999},
            artifacts={},
            baseline_comparison=None,
            error_message=str(exc),
            runtime_seconds=round(time.time() - started, 6),
        )

    valid, error_message = validate_experiment_config(family, config)
    if not valid:
        spec = _make_invalid_spec(family, normalized_config, data, experiment_id=experiment_id)
        return ExperimentResult(
            spec=spec,
            status="invalid",
            objective_score=None,
            metrics={},
            robustness={"viable": False, "negative_windows": 999},
            artifacts={},
            baseline_comparison=None,
            error_message=error_message,
            runtime_seconds=round(time.time() - started, 6),
        )

    spec = _make_experiment_spec(family.strip().lower(), normalized_config, data, experiment_id=experiment_id)

    if persist and has_experiment_result(spec.config_hash, family=spec.family, base_dir=base_dir):
        loaded = load_experiment_result(spec.config_hash, family=spec.family, base_dir=base_dir)
        if loaded is not None:
            duplicate = _result_from_loaded(loaded, status="duplicate")
            baseline_comparison = duplicate.baseline_comparison
            if baseline_comparison is None and compare_to_baseline:
                baseline_comparison = _build_baseline_comparison(
                    spec,
                    duplicate.metrics,
                    duplicate.objective_score,
                    duplicate.robustness,
                    compare_to_baseline,
                )
            return ExperimentResult(
                spec=spec,
                status="duplicate",
                objective_score=duplicate.objective_score,
                metrics=duplicate.metrics,
                robustness=duplicate.robustness,
                artifacts=duplicate.artifacts,
                baseline_comparison=baseline_comparison,
                error_message=None,
                runtime_seconds=round(time.time() - started, 6),
            )

    try:
        weights = build_strategy_weights_for_experiment(spec.family, data, normalized_config)
        raw_metrics = run_backtest(weights, data)
        metrics = _derive_extended_metrics(raw_metrics)
        objective_score = evaluate_objective(metrics, spec.family)
        robustness = evaluate_robustness(metrics)
        baseline_comparison = _build_baseline_comparison(
            spec,
            metrics,
            objective_score,
            robustness,
            compare_to_baseline,
        )
        runtime_seconds = round(time.time() - started, 6)

        if float(metrics.get("trades_per_year", 0.0) or 0.0) <= 0.0:
            status = "no_trades"
        else:
            status = "success"

        result = ExperimentResult(
            spec=spec,
            status=status,
            objective_score=objective_score,
            metrics=metrics,
            robustness=robustness,
            artifacts={},
            baseline_comparison=baseline_comparison,
            error_message=None,
            runtime_seconds=runtime_seconds,
        )
    except Exception as exc:
        result = ExperimentResult(
            spec=spec,
            status="error",
            objective_score=None,
            metrics={},
            robustness={"viable": False, "negative_windows": 999},
            artifacts={},
            baseline_comparison=None,
            error_message=str(exc),
            runtime_seconds=round(time.time() - started, 6),
        )

    if persist:
        payload = asdict(result)
        payload["spec"] = asdict(result.spec)
        save_experiment_result(payload, base_dir=base_dir)
    return result


def run_experiment_batch(
    family: str,
    specs: list[ExperimentSpec],
    data: dict[str, Any] | None = None,
    base_dir: str = "experiments",
) -> list[ExperimentResult]:
    data = data or load_data()
    results = []
    for spec in specs:
        result = run_single_experiment(
            family=spec.family,
            config=spec.params,
            data=data,
            persist=True,
            base_dir=base_dir,
            experiment_id=spec.experiment_id or None,
        )
        results.append(result)
    return results
