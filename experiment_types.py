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
    strategy_type: str | None = None
    source_type: str | None = None
    template_id: str | None = None
    hypothesis: str | None = None
    reason_selected: str | None = None
    novelty_score: float | None = None
    exploration_mode: str | None = None
    proposal_role: str | None = None
    region_label: str | None = None
    duplicate_risk: str | None = None
    dead_zone_risk: float | None = None
    parent_config_hash: str | None = None
    near_duplicate_of: str | None = None
    dead_zone_flags: list[str] | None = None
    selection_score: float | None = None
    source_proposal_id: str | None = None
    source_batch_id: str | None = None
    source_idea_ids: list[str] | None = None

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
    max_workers: int | None = None
    fail_fast: bool = False
    execution_mode: str = "auto"
    include_filters: dict[str, Any] | None = None
    exclude_filters: dict[str, Any] | None = None
    precomputed_configs: dict[str, list[dict[str, Any]]] | None = None
    precomputed_specs: dict[str, list[ExperimentSpec]] | None = None
    source_proposal_id: str | None = None
    proposal_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatchResult:
    request: BatchRequest
    status: str
    total_sampled: int
    total_executed: int
    total_skipped: int
    total_failed: int
    results: list[ExperimentResult]
    max_workers: int | None = None
    execution_mode: str = "sequential"
    worker_failures: int = 0
    leaderboard_path: str | None = None
    raw_results_path: str | None = None
    summary_path: str | None = None


@dataclass(frozen=True)
class ProposalRequest:
    proposal_id: str
    timestamp_utc: str
    source_batch_ids: list[str]
    strategy_families: list[str]
    objective_name: str = "wf_v1_score"
    baseline_name: str | None = None
    seed: int = 42
    exploration_fraction: float = 0.65
    exploitation_fraction: float = 0.35
    max_experiments: int = 20
    per_family_budgets: dict[str, int] | None = None
    resume: bool = True
    novelty_floor: float = 0.15
    template_fraction: float = 0.50
    cross_family_fraction: float = 0.20
    max_near_duplicate_distance: int = 1
    stagnation_escape_batches: int = 3
    allow_external_seeds: bool = False
    source_idea_ids: list[str] | None = None
    use_idea_queue: bool = True
    use_analysis_guidance: bool = True
    min_viable_fill_rate: float = 0.50
    min_viable_candidates: int | None = None
    large_search_threshold: int = 50
    min_large_search_candidates: int = 48
    quality_gate: bool = True
    persist_scorecards: bool = True
    persist_memory: bool = True
    persist_proposal: bool = True


@dataclass(frozen=True)
class ProposalResult:
    request: ProposalRequest
    status: str
    candidate_configs: dict[str, list[dict[str, Any]]]
    reasoning_summary: dict[str, Any]
    candidate_metadata: dict[str, list[dict[str, Any]]] | None = None
    proposal_path: str | None = None
    summary_path: str | None = None
