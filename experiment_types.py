from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    params: dict[str, Any]
    search_method: str = "single"
    objective_name: str = "wf_v1_score"
    batch_id: str = "single"
    config_hash: str = ""
    experiment_id: str = ""
    timestamp_utc: str = ""
    benchmark_source: str | None = None
    dataset_id: str | None = None
    data_start: str | None = None
    data_end: str | None = None
    split: str = "walk-forward"
    git_commit: str | None = None
    family_version: str | None = None

    @property
    def strategy_family(self) -> str:
        return self.family

    @property
    def config(self) -> dict[str, Any]:
        return self.params


@dataclass(frozen=True)
class ExperimentResult:
    spec: ExperimentSpec
    status: str
    objective_score: float | None
    metrics: dict[str, Any]
    robustness: dict[str, Any]
    artifacts: dict[str, str]
    baseline_comparison: dict[str, Any] | None = None
    error_message: str | None = None
    runtime_seconds: float | None = None

    @property
    def score(self) -> float | None:
        return self.objective_score

    @property
    def error(self) -> str | None:
        return self.error_message


@dataclass(frozen=True)
class BatchRequest:
    batch_id: str
    timestamp_utc: str
    strategy_families: list[str]
    sampler_type: str = "random"
    max_experiments: int = 20
    max_per_family: int = 20
    seed: int = 42
    persist: bool = True
    resume: bool = True
    objective_name: str = "wf_v1_score"
    include_filters: dict[str, Any] | None = None
    exclude_filters: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchResult:
    request: BatchRequest
    status: str
    total_sampled: int
    total_executed: int
    total_skipped: int
    total_failed: int
    results: list[ExperimentResult]
    leaderboard_path: str | None = None
    raw_results_path: str | None = None
    summary_path: str | None = None
