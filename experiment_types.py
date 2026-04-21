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
    source_region_id: str | None = None
    source_grid_search_id: str | None = None
    source_cells: list[dict[str, Any]] | None = None
    region_class: str | None = None
    promotion_recommendation: str | None = None
    region_state: str | None = None
    neighbor_of_cell: dict[str, Any] | None = None
    neighbor_generation_reason: str | None = None
    neighbor_grid_status: dict[str, str] | None = None
    validator_mode: str | None = None
    allowed_override_params: list[str] | None = None
    allowed_override_values: dict[str, list[Any]] | None = None
    off_grid_params: list[str] | None = None
    validation_override_reason: str | None = None
    source_idea_ids: list[str] | None = None
    idea_id: str | None = None
    idea_source: str | None = None
    idea_kind: str | None = None
    novelty_reason: str | None = None
    is_new_idea: bool = False
    is_structurally_novel: bool = False
    is_uncommon_idea: bool = False
    is_branch_repeat: bool = False
    repeat_branch_flag: bool = False
    repeat_branch_depth: int | None = None
    new_idea_budget_bucket: str | None = None
    uncommon_idea_reason: str | None = None
    synthesized_template_family: str | None = None
    synthesis_rationale: str | None = None
    confirmation_state: str | None = None
    confirmation_required: bool = False
    confirmation_reason: str | None = None
    confirmation_batch_id: str | None = None
    confirmation_trial_kind: str | None = None
    targeted_follow_up_required: bool = False
    targeted_follow_up_reason: str | None = None
    targeted_follow_up_type: str | None = None
    targeted_follow_up_priority: float | None = None
    targeted_follow_up_batch_id: str | None = None
    holdout_check_required: bool = False
    holdout_check_type: str | None = None
    holdout_check_status: str | None = None
    holdout_check_outcome: str | None = None
    holdout_check_scope: str | None = None
    holdout_check_batch_id: str | None = None
    holdout_horizon_tags: list[str] | None = None
    holdout_regime_tags: list[str] | None = None
    branch_budgets: dict[str, list[dict[str, Any]]] | None = None
    branch_budget_rationale: dict[str, Any] | None = None

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
    new_idea_budget: int | None = None
    refinement_budget: int | None = None
    confirmation_budget: int | None = None
    new_idea_quota: int | None = None
    uncommon_idea_quota: int | None = None
    uncommon_idea_budget: int | None = None
    structural_idea_budget: int | None = None
    out_of_box_budget: int | None = None
    max_routine_template_budget: int | None = None
    repeat_branch_cap: int | None = None
    max_same_template_per_cycle: int | None = None
    max_same_lineage_per_cycle: int | None = None
    structural_novelty_threshold: float = 0.55
    uncommon_template_bonus: float = 0.10
    structural_execution_lane_enabled: bool = True
    structural_execution_lane_budget: int | None = None
    structural_execution_lane_min_candidates: int | None = None
    structural_execution_lane_repeat_cap: int | None = None
    branch_budgets: dict[str, list[dict[str, Any]]] | None = None
    branch_budget_rationale: dict[str, Any] | None = None
    use_idea_queue: bool = True
    use_analysis_guidance: bool = True
    history_limit_per_family: int | None = None
    confirmation_state: str | None = None
    confirmation_required: bool = False
    confirmation_reason: str | None = None
    confirmation_batch_id: str | None = None
    confirmation_focus_family: str | None = None
    confirmation_batch_experiments: int | None = None
    confirmation_outcome: str | None = None
    targeted_follow_up_required: bool = False
    targeted_follow_up_reason: str | None = None
    targeted_follow_up_type: str | None = None
    targeted_follow_up_priority: float = 0.0
    targeted_follow_up_batch_id: str | None = None
    holdout_check_required: bool = False
    holdout_check_type: str | None = None
    holdout_check_status: str | None = None
    holdout_check_outcome: str | None = None
    holdout_check_scope: str | None = None
    holdout_check_batch_id: str | None = None
    holdout_horizon_tags: list[str] | None = None
    holdout_regime_tags: list[str] | None = None
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


@dataclass(frozen=True)
class RuntimeDecisionInput:
    workspace_root: str = "."
    experiments_dir: str = "experiments"
    strategy_families: list[str] | None = None
    max_experiments: int = 24
    seed: int = 42
    exploration_fraction: float = 0.65
    exploitation_fraction: float = 0.35
    large_search_threshold: int = 50
    min_large_search_candidates: int = 48
    stagnation_escape_batches: int = 3


@dataclass(frozen=True)
class RuntimeDecision:
    decision_id: str
    timestamp_utc: str
    status: str
    selected_families: list[str]
    cycle_mode: str
    max_experiments: int
    exploration_fraction: float
    exploitation_fraction: float
    family_budgets: dict[str, int] | None
    large_search_mode: bool
    min_large_search_candidates: int
    dashboard_report_id: str | None
    latest_batch_overview: dict[str, Any] | None
    latest_non_empty_batch: dict[str, Any] | None
    best_overall: dict[str, Any] | None
    best_viable: dict[str, Any] | None
    best_baseline_beating: dict[str, Any] | None
    family_scorecards: dict[str, dict[str, Any]]
    lineage_summary: dict[str, Any] | None
    used_signals: dict[str, Any]
    rationale: dict[str, Any]
    branch_budgets: dict[str, list[dict[str, Any]]] | None = None
    branch_budget_rationale: dict[str, Any] | None = None
    new_idea_budget: int | None = None
    refinement_budget: int | None = None
    confirmation_budget: int | None = None
    new_idea_quota: int | None = None
    uncommon_idea_quota: int | None = None
    repeat_branch_cap: int | None = None
    max_same_template_per_cycle: int | None = None
    max_same_lineage_per_cycle: int | None = None
    structural_novelty_threshold: float = 0.55
    uncommon_template_bonus: float = 0.10
    promotion_state: str = "unconfirmed"
    winner_family: str | None = None
    winner_config_hash: str | None = None
    winner_experiment_id: str | None = None
    winner_promotion_status: str = "not_promoted"
    winner_exploitation_cap: float | None = None
    winner_validation_horizon_tags: list[str] | None = None
    winner_validation_regime_tags: list[str] | None = None
    winner_validation_scope: str | None = None
    winner_validation_confidence: float | None = None
    winner_validation_coverage: float | None = None
    winner_validation_needs_follow_up: bool = False
    targeted_follow_up_required: bool = False
    targeted_follow_up_reason: str | None = None
    targeted_follow_up_type: str | None = None
    targeted_follow_up_priority: float | None = None
    targeted_follow_up_batch_id: str | None = None
    holdout_check_required: bool = False
    holdout_check_type: str | None = None
    holdout_check_status: str | None = None
    holdout_check_outcome: str | None = None
    holdout_check_scope: str | None = None
    holdout_check_batch_id: str | None = None
    holdout_horizon_tags: list[str] | None = None
    holdout_regime_tags: list[str] | None = None
    confirmation_batch_requested: bool = False
    confirmation_required: bool = False
    confirmation_reason: str | None = None
    confirmation_batch_id: str | None = None
    confirmation_outcome: str | None = None
    planned_max_experiments: int | None = None
    confirmation_family_budgets: dict[str, int] | None = None
    promotion_state_record: dict[str, Any] | None = None
    promotion_state_blocked_pending_new_evidence: bool = False
    promotion_state_changed_this_cycle: bool = False
    fallback_used: bool = False
    # Explicit holdout gate state — one of:
    #   no_winner | holdout_pending_long_horizon | holdout_pending_high_volatility
    #   | holdout_pending_combined | holdout_passed | holdout_failed
    #   | promotion_blocked | promotion_can_advance | generic_confirmation_pending
    holdout_rationale: str = "no_winner"
    holdout_rationale_detail: str | None = None
    structural_novelty_budget: int | None = None


@dataclass(frozen=True)
class PromotionStateRecord:
    family: str
    config_hash: str
    experiment_id: str | None = None
    promotion_state: str = "unconfirmed"
    winner_promotion_status: str = "not_promoted"
    confirmation_required: bool = False
    confirmation_reason: str | None = None
    confirmation_batch_id: str | None = None
    confirmation_outcome: str | None = None
    holdout_check_required: bool = False
    holdout_check_type: str | None = None
    holdout_check_status: str | None = None
    holdout_check_outcome: str | None = None
    holdout_check_scope: str | None = None
    holdout_check_batch_id: str | None = None
    confirmation_history: list[dict[str, Any]] | None = None
    holdout_history: list[dict[str, Any]] | None = None
    history: list[dict[str, Any]] | None = None
    last_confirmation_timestamp_utc: str | None = None
    last_confirmation_cycle_id: str | None = None
    last_failed_confirmation_timestamp_utc: str | None = None
    last_failed_confirmation_cycle_id: str | None = None
    last_holdout_timestamp_utc: str | None = None
    last_holdout_cycle_id: str | None = None
    blocked_pending_new_evidence: bool = False
    block_reason: str | None = None
    last_seen_timestamp_utc: str | None = None
    last_seen_cycle_id: str | None = None
    source_batch_ids: list[str] | None = None
    source_proposal_id: str | None = None
    winner_family: str | None = None
    validation_horizon_tags: list[str] | None = None
    validation_regime_tags: list[str] | None = None
    validation_scope: str | None = None
    validation_confidence: float | None = None
    validation_coverage: float | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class LineageRecord:
    family: str
    config_hash: str
    experiment_id: str | None = None
    parent_config_hash: str | None = None
    lineage_root_config_hash: str | None = None
    lineage_depth: int = 0
    lineage_type: str = "seed"
    descendant_count: int = 0
    confirmation_descendant_count: int = 0
    holdout_descendant_count: int = 0
    rejected_descendant_count: int = 0
    best_descendant_config_hash: str | None = None
    best_descendant_experiment_id: str | None = None
    best_descendant_objective_score: float | None = None
    best_descendant_status: str | None = None
    lineage_status_summary: str = "seed"
    lineage_trust_score: float = 0.0
    branch_balance: float = 0.0
    source_batch_ids: list[str] | None = None
    source_proposal_id: str | None = None
    history: list[dict[str, Any]] | None = None
    last_seen_timestamp_utc: str | None = None
    last_seen_cycle_id: str | None = None
    updated_at: str | None = None
