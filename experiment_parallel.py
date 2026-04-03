from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any

from experiment_runner import run_single_experiment
from experiment_types import ExperimentResult, ExperimentSpec
from prepare import load_data


_WORKER_DATA: dict[str, Any] | None = None
_WORKER_BASE_DIR: str = "experiments"
_WORKER_BASELINE_BY_FAMILY: dict[str, str | None] | None = None


def _spec_metadata(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "source_type": spec.source_type,
        "template_id": spec.template_id,
        "hypothesis": spec.hypothesis,
        "reason_selected": spec.reason_selected,
        "novelty_score": spec.novelty_score,
        "exploration_mode": spec.exploration_mode,
        "parent_config_hash": spec.parent_config_hash,
        "near_duplicate_of": spec.near_duplicate_of,
        "dead_zone_flags": spec.dead_zone_flags,
        "selection_score": spec.selection_score,
        "source_proposal_id": spec.source_proposal_id,
        "source_batch_id": spec.source_batch_id,
    }


def _worker_init(
    data: dict[str, Any],
    base_dir: str,
    baseline_by_family: dict[str, str | None] | None,
) -> None:
    global _WORKER_DATA, _WORKER_BASE_DIR, _WORKER_BASELINE_BY_FAMILY
    _WORKER_DATA = data
    _WORKER_BASE_DIR = base_dir
    _WORKER_BASELINE_BY_FAMILY = baseline_by_family or {}


def _empty_error_result(spec: ExperimentSpec, message: str) -> ExperimentResult:
    return ExperimentResult(
        spec=spec,
        status="error",
        objective_score=None,
        metrics={},
        robustness={"viable": False, "negative_windows": 999},
        artifacts={},
        baseline_comparison=None,
        error_message=message,
        runtime_seconds=None,
    )


def worker_execute_spec(
    spec: ExperimentSpec,
    *,
    data: dict[str, Any] | None = None,
    base_dir: str = "experiments",
    compare_to_baseline: str | None = None,
) -> ExperimentResult:
    source_data = data if data is not None else (_WORKER_DATA if _WORKER_DATA is not None else load_data())
    baseline_name = compare_to_baseline
    if baseline_name is None and _WORKER_BASELINE_BY_FAMILY is not None:
        baseline_name = _WORKER_BASELINE_BY_FAMILY.get(spec.family)

    try:
        return run_single_experiment(
            family=spec.family,
            config=spec.params,
            data=source_data,
            persist=False,
            base_dir=base_dir,
            experiment_id=spec.experiment_id or None,
            compare_to_baseline=baseline_name,
            idea_metadata=_spec_metadata(spec),
            spec=spec,
        )
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        return _empty_error_result(spec, f"worker_failure: {exc}")


def _resolve_max_workers(max_workers: int | None) -> int:
    if max_workers is None:
        return 1
    if max_workers <= 1:
        return 1
    if max_workers > 8:
        raise ValueError("max_workers must be <= 8.")
    return max_workers


def run_experiments_parallel(
    specs: list[ExperimentSpec],
    *,
    data: dict[str, Any] | None = None,
    base_dir: str = "experiments",
    max_workers: int | None = 6,
    fail_fast: bool = False,
    baseline_by_family: dict[str, str | None] | None = None,
) -> tuple[list[ExperimentResult], int]:
    if not specs:
        return [], 0

    worker_data = data if data is not None else load_data()
    effective_workers = _resolve_max_workers(max_workers)
    if effective_workers == 1:
        results = [
            worker_execute_spec(
                spec,
                data=worker_data,
                base_dir=base_dir,
                compare_to_baseline=(baseline_by_family or {}).get(spec.family),
            )
            for spec in specs
        ]
        return results, 0

    try:
        context = get_context("fork")
    except ValueError:  # pragma: no cover - non-Linux fallback
        context = get_context()

    results: list[ExperimentResult | None] = [None] * len(specs)
    worker_failures = 0
    with ProcessPoolExecutor(
        max_workers=effective_workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(worker_data, base_dir, baseline_by_family or {}),
    ) as executor:
        future_map = {
            executor.submit(worker_execute_spec, spec): (idx, spec)
            for idx, spec in enumerate(specs)
        }
        for future in as_completed(future_map):
            idx, spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - worker crash boundary
                worker_failures += 1
                result = _empty_error_result(spec, f"worker_exception: {exc}")
                if fail_fast:
                    for pending in future_map:
                        if pending != future:
                            pending.cancel()
                    results[idx] = result
                    break
            results[idx] = result
            if fail_fast and result.status == "error":
                for pending in future_map:
                    if pending != future:
                        pending.cancel()
                break

    return [result or _empty_error_result(spec, "missing_result") for result, spec in zip(results, specs, strict=False)], worker_failures
