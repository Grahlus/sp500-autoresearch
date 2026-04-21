from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from agents.schemas import load_idea_records, load_latest_analysis_report, update_idea_record_status, IDEA_ACTIVE_STATUSES
from experiment_idea_manager import select_hot_representatives
from experiment_batch import DEFAULT_BASELINES
from experiment_lineage import build_branch_budget_plan, build_lineage_summary
from experiment_idea_library import build_template_payload, expand_template_candidates, list_idea_templates, load_external_idea_seeds
from experiment_idea_yield import build_idea_yield_summary, load_latest_idea_yield_summary, save_idea_yield_summary
from experiment_memory import load_lineage_state_records, load_research_memory, save_research_memory, update_family_memory
from experiment_novelty import coarse_signature_key, score_candidate, signature_distance
from experiment_scorecards import build_family_scorecards, save_family_scorecards, scorecards_to_records
from experiment_spaces import (
    get_family_default_config,
    get_family_search_space,
    list_searchable_families,
    normalize_experiment_config,
    sample_random_candidates,
)
from experiment_search import smart_sample_candidates
from experiment_store import compute_config_hash, load_prior_results, load_results_index, save_proposal_result
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
    new_idea_budget: int | None = None,
    refinement_budget: int | None = None,
    confirmation_budget: int | None = None,
    new_idea_quota: int | None = None,
    uncommon_idea_quota: int | None = None,
    uncommon_idea_budget: int | None = None,
    structural_idea_budget: int | None = None,
    out_of_box_budget: int | None = None,
    max_routine_template_budget: int | None = None,
    repeat_branch_cap: int | None = None,
    max_same_template_per_cycle: int | None = None,
    max_same_lineage_per_cycle: int | None = None,
    structural_novelty_threshold: float = 0.55,
    uncommon_template_bonus: float = 0.10,
    structural_execution_lane_enabled: bool = True,
    structural_execution_lane_budget: int | None = None,
    structural_execution_lane_min_candidates: int | None = None,
    structural_execution_lane_repeat_cap: int | None = None,
    branch_budgets: dict[str, list[dict[str, Any]]] | None = None,
    branch_budget_rationale: dict[str, Any] | None = None,
    use_idea_queue: bool = True,
    use_analysis_guidance: bool = True,
    history_limit_per_family: int | None = None,
    confirmation_state: str | None = None,
    confirmation_required: bool = False,
    confirmation_reason: str | None = None,
    confirmation_batch_id: str | None = None,
    confirmation_focus_family: str | None = None,
    confirmation_batch_experiments: int | None = None,
    targeted_follow_up_required: bool = False,
    targeted_follow_up_reason: str | None = None,
    targeted_follow_up_type: str | None = None,
    targeted_follow_up_priority: float = 0.0,
    targeted_follow_up_batch_id: str | None = None,
    holdout_check_required: bool = False,
    holdout_check_type: str | None = None,
    holdout_check_status: str | None = None,
    holdout_check_outcome: str | None = None,
    holdout_check_scope: str | None = None,
    holdout_check_batch_id: str | None = None,
    holdout_horizon_tags: list[str] | None = None,
    holdout_regime_tags: list[str] | None = None,
    min_viable_fill_rate: float = 0.50,
    min_viable_candidates: int | None = None,
    large_search_threshold: int = 50,
    min_large_search_candidates: int = 48,
    quality_gate: bool = True,
    persist_scorecards: bool = True,
    persist_memory: bool = True,
    persist_proposal: bool = True,
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
        new_idea_budget=None if new_idea_budget is None else int(new_idea_budget),
        refinement_budget=None if refinement_budget is None else int(refinement_budget),
        confirmation_budget=None if confirmation_budget is None else int(confirmation_budget),
        new_idea_quota=None if new_idea_quota is None else int(new_idea_quota),
        uncommon_idea_quota=None if uncommon_idea_quota is None else int(uncommon_idea_quota),
        uncommon_idea_budget=None if uncommon_idea_budget is None else int(uncommon_idea_budget),
        structural_idea_budget=None if structural_idea_budget is None else int(structural_idea_budget),
        out_of_box_budget=None if out_of_box_budget is None else int(out_of_box_budget),
        max_routine_template_budget=None if max_routine_template_budget is None else int(max_routine_template_budget),
        repeat_branch_cap=None if repeat_branch_cap is None else int(repeat_branch_cap),
        max_same_template_per_cycle=None if max_same_template_per_cycle is None else int(max_same_template_per_cycle),
        max_same_lineage_per_cycle=None if max_same_lineage_per_cycle is None else int(max_same_lineage_per_cycle),
        structural_novelty_threshold=float(structural_novelty_threshold),
        uncommon_template_bonus=float(uncommon_template_bonus),
        structural_execution_lane_enabled=bool(structural_execution_lane_enabled),
        structural_execution_lane_budget=None if structural_execution_lane_budget is None else int(structural_execution_lane_budget),
        structural_execution_lane_min_candidates=None if structural_execution_lane_min_candidates is None else int(structural_execution_lane_min_candidates),
        structural_execution_lane_repeat_cap=None if structural_execution_lane_repeat_cap is None else int(structural_execution_lane_repeat_cap),
        branch_budgets=branch_budgets,
        branch_budget_rationale=branch_budget_rationale,
        use_idea_queue=bool(use_idea_queue),
        use_analysis_guidance=bool(use_analysis_guidance),
        history_limit_per_family=None if history_limit_per_family is None else int(history_limit_per_family),
        confirmation_state=confirmation_state,
        confirmation_required=bool(confirmation_required),
        confirmation_reason=confirmation_reason,
        confirmation_batch_id=confirmation_batch_id,
        confirmation_focus_family=confirmation_focus_family,
        confirmation_batch_experiments=None if confirmation_batch_experiments is None else int(confirmation_batch_experiments),
        targeted_follow_up_required=bool(targeted_follow_up_required),
        targeted_follow_up_reason=targeted_follow_up_reason,
        targeted_follow_up_type=targeted_follow_up_type,
        targeted_follow_up_priority=float(targeted_follow_up_priority),
        targeted_follow_up_batch_id=targeted_follow_up_batch_id,
        holdout_check_required=bool(holdout_check_required),
        holdout_check_type=holdout_check_type,
        holdout_check_status=holdout_check_status,
        holdout_check_outcome=holdout_check_outcome,
        holdout_check_scope=holdout_check_scope,
        holdout_check_batch_id=holdout_check_batch_id,
        holdout_horizon_tags=list(holdout_horizon_tags or []),
        holdout_regime_tags=list(holdout_regime_tags or []),
        min_viable_fill_rate=float(min_viable_fill_rate),
        min_viable_candidates=None if min_viable_candidates is None else int(min_viable_candidates),
        large_search_threshold=int(large_search_threshold),
        min_large_search_candidates=int(min_large_search_candidates),
        quality_gate=bool(quality_gate),
        persist_scorecards=bool(persist_scorecards),
        persist_memory=bool(persist_memory),
        persist_proposal=bool(persist_proposal),
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
    # Accept active statuses only (excludes archived/retired/merged/executed)
    filtered = [
        record
        for record in records
        if record.get("status", "new") in IDEA_ACTIVE_STATUSES
        and (record.get("family") in request.strategy_families or record.get("family") is None)
    ]
    # Apply cluster-representative selection so that near-duplicate ideas do not
    # all compete equally; each cluster sends at most max_per_cluster representatives.
    filtered = select_hot_representatives(filtered)
    filtered.sort(
        key=lambda record: (
            -float(record.get("priority") or 0.0),
            -float(record.get("novelty_score") or 0.0),
            not bool(record.get("is_new_idea")),
            bool(record.get("is_branch_repeat")),
            str(record.get("idea_id") or ""),
        )
    )
    # Mark selected ideas as shortlisted so lifecycle tracking is accurate.
    for record in filtered:
        idea_id = str(record.get("idea_id") or "")
        if idea_id and record.get("status", "new") == "new":
            update_idea_record_status(idea_id, "shortlisted", workspace_root)
    return filtered


def _idea_signature_from_payload(payload: dict[str, Any]) -> str:
    keys = (
        "family",
        "source",
        "idea_source",
        "idea_kind",
        "template_id",
        "idea_id",
        "synthesized_template_family",
    )
    canonical = {key: payload.get(key) for key in keys if payload.get(key) is not None}
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _structural_synthesis_family(idea: dict[str, Any]) -> str:
    family = str(idea.get("template_similarity_class") or idea.get("synthesized_template_family") or "").strip().lower()
    if family:
        return family
    idea_kind = str(idea.get("idea_kind") or "").strip().lower()
    mapping = {
        "new_portfolio_exposure_control": "portfolio_overlay",
        "new_regime_filter_concept": "regime_filter",
        "new_ranking_concept": "ranking_rethink",
        "new_per_stock_state_model": "state_model",
        "new_position_sizing_concept": "position_sizing",
        "new_cross_family_borrowing_concept": "cross_family_mechanism",
        "new_ml_architecture_family": "architecture_family",
        "new_rl_policy_concept": "policy_family",
        "new_crash_defense_mechanism": "crash_defense",
        "new_breadth_dispersion_market_state_concept": "breadth_filter",
        "literature_inspired_structural_template": "literature_transfer",
    }
    return mapping.get(idea_kind, "structural_extension")


def _synthesized_structural_config(family: str, idea: dict[str, Any], idea_yield: dict[str, Any]) -> dict[str, Any]:
    synth_family = _structural_synthesis_family(idea)
    config = get_family_default_config(family)
    state = str(idea_yield.get("idea_state") or "untested").strip().lower()
    quality = float(idea_yield.get("idea_quality_score") or 0.0)
    out_of_box = bool(idea.get("is_out_of_box"))
    uncommon = bool(idea.get("is_uncommon_idea"))

    if family == "momentum":
        overrides: dict[str, Any] = {
            "LOOKBACK_WEEKS": 39 if synth_family in {"state_model", "ranking_rethink"} else 26,
            "SKIP_WEEKS": 4 if synth_family in {"regime_filter", "crash_defense"} else 3,
            "REBAL_WEEKS": 2 if synth_family in {"portfolio_overlay", "position_sizing", "cross_family_mechanism"} else 4,
            "TOP_PCT": 0.015 if synth_family in {"ranking_rethink", "cross_family_mechanism"} else 0.025,
            "MA_WEEKS": 24 if synth_family in {"state_model", "regime_filter"} else 20,
            "STOP_TYPE": "adaptive",
            "STOP_LOSS_PCT": 0.15 if synth_family in {"crash_defense", "portfolio_overlay"} else 0.20,
            "STOP_PARABOLIC": 0.35 if synth_family in {"crash_defense", "regime_filter"} else 0.30,
            "FG_MIN": 30.0 if synth_family in {"regime_filter", "breadth_filter", "crash_defense"} else 22.0,
            "EXIT_PCT_RANK": 0.94 if synth_family in {"ranking_rethink", "cross_family_mechanism", "portfolio_overlay"} else 0.97,
            "RANK_EXIT_CONFIRM": 1 if synth_family in {"ranking_rethink", "cross_family_mechanism", "portfolio_overlay"} else None,
        }
        if synth_family == "portfolio_overlay":
            overrides["MIN_HOLD_DAYS"] = 10
        elif synth_family == "state_model":
            overrides["MIN_HOLD_DAYS"] = 15
        elif synth_family == "position_sizing":
            overrides["MIN_HOLD_DAYS"] = 10
        elif synth_family == "crash_defense":
            overrides["MIN_HOLD_DAYS"] = 15
    elif family == "superstock":
        overrides = {
            "max_positions": 3 if synth_family in {"portfolio_overlay", "crash_defense"} else 5,
            "price_min": 5.0,
            "price_max": 20.0 if synth_family in {"ranking_rethink", "cross_family_mechanism"} else 15.0,
            "min_dollar_volume": 2_000_000.0 if synth_family in {"portfolio_overlay", "breadth_filter"} else 1_000_000.0,
            "rs_rank_26w_min": 0.80 if synth_family in {"ranking_rethink", "state_model"} else 0.70,
            "rs_rank_52w_min": 0.80 if synth_family in {"ranking_rethink", "state_model"} else 0.70,
            "base_depth_max": 0.40 if synth_family in {"state_model", "ranking_rethink"} else 0.60,
            "weekly_range_median_max": 0.15 if synth_family in {"crash_defense", "breadth_filter"} else 0.18,
            "weekly_volatility_max": 0.10 if synth_family in {"crash_defense", "breadth_filter"} else 0.12,
            "volume_dryup_max": 0.80,
            "vix_hard_cap": 30.0 if synth_family in {"crash_defense", "regime_filter"} else 35.0,
            "vix_ma_multiplier": 1.10 if synth_family in {"crash_defense", "regime_filter"} else 1.25,
            "breakout_extension_max": 0.08 if synth_family in {"ranking_rethink", "state_model"} else 0.10,
            "daily_volume_expansion_mult": 2.00 if synth_family in {"position_sizing", "cross_family_mechanism"} else 1.50,
            "daily_dollar_volume_expansion_mult": 2.00 if synth_family in {"position_sizing", "cross_family_mechanism"} else 1.50,
            "parabolic_from_pivot_min": 0.30 if synth_family in {"cross_family_mechanism", "ranking_rethink"} else 0.25,
            "parabolic_above_10w_min": 0.25 if synth_family in {"cross_family_mechanism", "ranking_rethink"} else 0.20,
            "late_stage_range_mult": 1.50 if synth_family in {"breadth_filter", "crash_defense"} else 1.75,
            "late_stage_volume_mult": 1.50 if synth_family in {"breadth_filter", "crash_defense"} else 2.00,
        }
    elif family == "ml_ranker":
        overrides = {
            "model_type": "hgb" if synth_family in {"architecture_family", "state_model"} else "ridge",
            "lookback_days": 504 if synth_family in {"architecture_family", "state_model"} else 252,
            "horizon_days": 20 if synth_family in {"state_model", "ranking_rethink"} else 10,
            "rebalance_days": 10 if synth_family in {"portfolio_overlay", "policy_family"} else 5,
            "top_pct": 0.025 if synth_family in {"ranking_rethink", "state_model"} else 0.03,
            "max_positions": 20 if synth_family in {"architecture_family", "state_model"} else 10,
            "feature_set": "full" if synth_family in {"architecture_family", "state_model"} else "trend_volume",
            "allow_short": False,
            "use_vix_gate": True,
            "use_fear_greed_gate": bool(out_of_box or uncommon or quality >= 0.45 or state in {"promising", "active"}),
        }
    elif family == "rl_bandit":
        overrides = {
            "policy_type": "epsilon_greedy" if synth_family in {"policy_family", "cross_family_mechanism"} else "ucb",
            "lookback_days": 126 if synth_family in {"policy_family", "state_model"} else 252,
            "rebalance_days": 10 if synth_family in {"policy_family", "portfolio_overlay"} else 5,
            "epsilon": 0.20 if synth_family in {"policy_family", "cross_family_mechanism"} else 0.10,
            "ucb_bonus": 0.5 if synth_family in {"policy_family", "cross_family_mechanism"} else 1.0,
            "max_positions": 8 if synth_family in {"portfolio_overlay", "position_sizing"} else 5,
            "momentum_top_pct": 0.025 if synth_family in {"cross_family_mechanism", "ranking_rethink"} else 0.03,
            "superstock_top_pct": 0.025 if synth_family in {"cross_family_mechanism", "ranking_rethink"} else 0.03,
            "use_vix_gate": True,
            "use_fear_greed_gate": True,
        }
    else:
        overrides = {}

    if state == "promising" and family in {"ml_ranker", "rl_bandit"}:
        overrides["rebalance_days"] = overrides.get("rebalance_days", config.get("rebalance_days", 5))
    elif state == "retired":
        if "use_fear_greed_gate" in config:
            overrides["use_fear_greed_gate"] = True

    normalized = normalize_experiment_config(family, {**config, **overrides})
    return normalized


def _synthesized_structural_candidates(
    *,
    family: str,
    ideas: list[dict[str, Any]],
    idea_yield: dict[str, Any],
    recent_idea_signatures: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idea in ideas:
        idea_id = str(idea.get("idea_id") or "").strip()
        if not idea_id:
            continue
        synth_family = _structural_synthesis_family(idea)
        template_id = f"synth_structural_{family}_{synth_family}"
        config = _synthesized_structural_config(family, idea, idea_yield)
        signature = _idea_signature_from_payload(
            {
                "family": family,
                "idea_id": idea_id,
                "idea_source": idea.get("idea_source"),
                "source": "structural_synthesis",
                "idea_kind": idea.get("idea_kind"),
                "template_id": template_id,
                "synthesized_template_family": synth_family,
            }
        )
        if signature in recent_idea_signatures:
            continue
        rationale = (
            f"Synthesized from {idea_id} as {synth_family}; "
            f"idea_kind={idea.get('idea_kind')}; source_type=structural_synthesis; "
            f"idea_source={idea.get('idea_source')}; structural_distance={idea.get('structural_distance')}; "
            f"novelty_reason={idea.get('novelty_reason') or idea.get('uncommon_idea_reason')}"
        )
        candidates.append(
            {
                "config": config,
                "metadata": {
                    "idea_id": idea_id,
                    "source_type": "structural_synthesis",
                    "idea_source": idea.get("idea_source") or idea.get("source") or "unknown",
                    "idea_kind": idea.get("idea_kind") or "new_strategy_logic_concept",
                    "template_id": template_id,
                    "synthesized_template_family": synth_family,
                    "synthesis_rationale": rationale,
                    "structural_synthesis_family": synth_family,
                    "hypothesis": idea.get("hypothesis"),
                    "reason_selected": rationale,
                    "exploration_mode": "structural_synthesis",
                    "proposal_role": "explore",
                    "region_label": synth_family,
                    "source_idea_ids": [idea_id],
                    "template_tags": [synth_family, str(idea.get("idea_kind") or "structural")],
                    "is_structurally_novel": bool(idea.get("is_structurally_novel")),
                    "is_out_of_box": bool(idea.get("is_out_of_box")),
                    "is_uncommon_idea": bool(idea.get("is_uncommon_idea")),
                    "structural_novelty_score": float(idea.get("structural_distance") or 0.0),
                    "novelty_reason": idea.get("novelty_reason") or idea.get("uncommon_idea_reason") or rationale,
                    "uncommon_idea_reason": idea.get("uncommon_idea_reason") or rationale,
                },
                "idea": idea,
            }
        )
    candidates.sort(
        key=lambda item: (
            -int(bool(item["metadata"].get("is_out_of_box"))),
            -int(bool(item["metadata"].get("is_structurally_novel"))),
            -float(item["metadata"].get("structural_novelty_score") or 0.0),
            str(item["metadata"].get("idea_id") or ""),
        )
    )
    return candidates[:limit]


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


def _load_detailed_history(
    family: str,
    base_dir: str,
    *,
    limit: int | None = None,
    top_n_by_score: int = 20,
) -> pd.DataFrame:
    index = load_prior_results(family=family, base_dir=base_dir)
    rows: list[dict[str, Any]] = []
    if index.empty:
        return pd.DataFrame()

    if limit is not None and limit > 0 and len(index) > limit:
        if "timestamp_utc" in index.columns:
            recent = index.sort_values("timestamp_utc", na_position="first").tail(limit)
        else:
            recent = index.tail(limit)
        # Also include the top-scored rows so confirmation can target the best config,
        # not just the most recently run one (which may be a stale repeated run).
        if top_n_by_score > 0 and "objective_score" in index.columns:
            _score_col = pd.to_numeric(index["objective_score"], errors="coerce")
            top_scored = (
                index.assign(_score_sort=_score_col)
                .sort_values("_score_sort", ascending=False, na_position="last")
                .drop(columns=["_score_sort"])
                .head(top_n_by_score)
            )
            index = pd.concat([recent, top_scored]).drop_duplicates(
                subset=["experiment_id"]
            )
        else:
            index = recent

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
        try:
            from experiment_spaces import normalize_experiment_config
            normalize_experiment_config(family, config)
        except (ValueError, KeyError):
            continue
        entry = {
            "experiment_id": spec.get("experiment_id"),
            "batch_id": spec.get("batch_id"),
            "family": spec.get("strategy_family", spec.get("family")),
            "config_hash": spec.get("config_hash"),
            "config": config,
            "template_id": spec.get("template_id"),
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
            try:
                poor.add(coarse_signature_key(family, config))
            except (ValueError, KeyError):
                pass
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
        records.append(
            {
                "experiment_id": record.get("experiment_id"),
                "batch_id": record.get("batch_id"),
                "config_hash": record.get("config_hash"),
                "config": config,
                "objective_score": record.get("objective_score"),
                "status": record.get("status"),
                "viable": record.get("viable"),
                "comparison_status": record.get("comparison_status"),
                "beats_baseline_objective": record.get("beats_baseline_objective"),
                "delta_sharpe": record.get("delta_sharpe"),
                "template_id": record.get("template_id"),
                "strategy_type": record.get("strategy_type"),
                "source_type": record.get("source_type"),
                "family": record.get("family"),
            }
        )
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
        try:
            signature_map[config_hash] = coarse_signature_key(family, config)
        except (ValueError, KeyError):
            pass
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


def _template_cycle_key(template_id: str | None, source_type: str | None) -> str:
    return str(template_id or source_type or "unspecified")


def _structural_novelty_profile(
    *,
    family: str,
    source_type: str,
    strategy_type: str | None,
    template_id: str | None,
    template_tags: list[str] | None,
    template_counts: dict[str, int] | None,
    is_branch_repeat: bool,
    structural_novelty_threshold: float,
    uncommon_template_bonus: float,
) -> dict[str, Any]:
    template_key = _template_cycle_key(template_id, source_type)
    template_count = int((template_counts or {}).get(template_key, 0) or 0)
    source = str(source_type or "")
    strategy = str(strategy_type or "")
    tags = {str(tag) for tag in (template_tags or [])}
    structural_score = 0.0
    structural_reasons: list[str] = []

    if source in {"cross_family_hybrid", "external_seed", "idea_seed", "model_based", "policy_learning"}:
        structural_score += 0.55
        structural_reasons.append(f"source_type={source}")
    if source == "structural_synthesis":
        structural_score += 0.80
        structural_reasons.append("source_type=structural_synthesis")
    if source == "template_expansion" and template_key not in {"lineage_branch", "local_refinement"}:
        structural_score += 0.40
        structural_reasons.append("template_expansion")
    if strategy in {"ml", "rl"} and source not in {"confirmation_reproduction", "confirmation_check", "holdout_check"}:
        structural_score += 0.40
        structural_reasons.append(f"strategy_type={strategy}")
    if family != "momentum" and source in {"template_expansion", "cross_family_hybrid", "model_based", "policy_learning", "external_seed"}:
        structural_score += 0.25
        structural_reasons.append(f"non_primary_family={family}")
    if template_id and template_count <= 1 and source not in {"local_refinement", "confirmation_reproduction", "confirmation_check", "holdout_check"}:
        structural_score += 0.20 + max(0.0, uncommon_template_bonus)
        structural_reasons.append("underused_template")
    if tags & {"regime", "horizon", "sizing", "exit", "ranking", "ml", "rl"}:
        structural_score += 0.15
        structural_reasons.append(f"concept_tags={sorted(tags)}")

    # Template density: reward templates under-represented relative to median run share.
    if (
        template_counts
        and template_id
        and source not in {"local_refinement", "confirmation_reproduction", "confirmation_check", "holdout_check"}
        and not is_branch_repeat
    ):
        total_template_runs = sum(template_counts.values()) or 1
        template_run_share = template_count / total_template_runs
        n_templates = max(1, len(template_counts))
        median_run_share = 1.0 / n_templates
        if template_run_share < median_run_share * 0.5:
            structural_score += 0.18
            structural_reasons.append(f"template_density_below_median(share={template_run_share:.3f})")
        elif template_run_share > median_run_share * 2.5 and structural_score < 0.55:
            structural_score -= 0.08
            structural_reasons.append(f"template_density_above_2.5x_median(share={template_run_share:.3f})")

    # Generic cross-family/cross-template borrowing bonus beyond the base source_type bonus.
    if source == "cross_family_hybrid" or (tags & {"cross_family", "cross_template"}):
        structural_score += 0.12
        structural_reasons.append("cross_family_borrowing_bonus")

    if is_branch_repeat:
        structural_score = min(structural_score, 0.30)
        structural_reasons.append("branch_repeat_cap")

    structural_score = max(0.0, min(1.0, structural_score))
    is_structural = bool(structural_score >= max(0.0, min(1.0, structural_novelty_threshold)))
    is_uncommon = bool(
        is_structural
        and (
            template_count <= 1
            or strategy in {"ml", "rl"}
            or family != "momentum"
            or source in {"cross_family_hybrid", "external_seed", "model_based", "policy_learning", "structural_synthesis"}
        )
    )
    uncommon_reason = None
    if is_uncommon:
        uncommon_reason = "; ".join(structural_reasons or ["underused structural idea"])
    return {
        "is_structurally_novel": is_structural,
        "is_out_of_box": bool(structural_score >= 0.75),
        "is_uncommon_idea": is_uncommon,
        "structural_novelty_score": round(structural_score, 6),
        "structural_novelty_reason": "; ".join(structural_reasons) if structural_reasons else "parameter-level variation",
        "uncommon_idea_reason": uncommon_reason,
    }


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



def _get_novelty_floor(
    exploration_mode: str,
    *,
    stagnation_batches: int = 0,
    is_confirmation: bool = False,
    is_holdout: bool = False,
    base_floor: float = 0.15,
) -> float:
    """Per-mode novelty floor. Confirmation/holdout return 0.0 to preserve small-batch bypass."""
    if is_confirmation or is_holdout:
        return 0.0
    if stagnation_batches >= 2:
        return max(0.05, base_floor * 0.50)
    mode = str(exploration_mode or "").strip().lower()
    if mode in {"stagnation_escape", "saturation_escape", "structural_exploration"}:
        return max(0.05, base_floor * 0.40)
    if mode in {"broader_exploration", "large_search", "idea_seed"}:
        return max(0.08, base_floor * 0.65)
    if mode in {"local_refinement", "branch_refinement"}:
        return max(0.10, base_floor)
    return base_floor


def _adaptive_budget_fractions(
    *,
    cycle_mode: str,
    stagnation_batches: int,
    viable_rate: float,
    base_template_fraction: float,
    base_cross_family_fraction: float,
    branch_dominance: float,
    has_multiple_families: bool,
) -> tuple[float, float]:
    """Return adapted (template_fraction, cross_family_fraction) for this cycle."""
    t_frac = float(base_template_fraction)
    cf_frac = float(base_cross_family_fraction)
    if stagnation_batches >= 2:
        t_frac = min(0.65, t_frac * 1.25)
        if has_multiple_families:
            cf_frac = min(0.30, cf_frac * 1.30)
    if branch_dominance >= 0.50:
        t_frac = min(0.65, t_frac * 1.15)
    if viable_rate < 0.10:
        t_frac = min(0.65, t_frac + 0.05)
    if cycle_mode == "large-search":
        t_frac = min(0.65, max(t_frac, 0.40))
    if stagnation_batches >= 2 and cycle_mode not in {"confirmation", "large-search"}:
        cf_frac = min(0.30, max(cf_frac, 0.18)) if has_multiple_families else cf_frac
    return round(t_frac, 4), round(cf_frac, 4)

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


def _repair_local_neighbor_candidate(family: str, candidate: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(candidate)
    if family == "momentum":
        stop_type = repaired.get("STOP_TYPE")
        if stop_type in {"fixed", "adaptive"} and repaired.get("STOP_LOSS_PCT") is None:
            repaired["STOP_LOSS_PCT"] = get_family_search_space(family)["STOP_LOSS_PCT"]["default"]
        if stop_type == "adaptive" and repaired.get("STOP_PARABOLIC") is None:
            repaired["STOP_PARABOLIC"] = get_family_search_space(family)["STOP_PARABOLIC"]["default"]
    return repaired


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
        candidate = _repair_local_neighbor_candidate(family, candidate)
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
    cycle_mode: str = "normal_exploration",
    history_count: int = 0,
    stagnation_batches: int = 0,
    top_configs: list[dict[str, Any]] | None = None,
    out_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Sample exploration candidate configs using smart, non-exhaustive search.

    Automatically selects coarse_grid / latin_hypercube / local_refinement / hybrid
    based on cycle context. Pass out_metadata={} to receive search diagnostics.
    """
    dead_zones = dead_zones or {}
    candidates, search_meta = smart_sample_candidates(
        family,
        n=limit,
        seed=seed,
        cycle_mode=cycle_mode,
        history_count=history_count,
        stagnation_batches=stagnation_batches,
        top_configs=top_configs,
        explored_hashes=explored_hashes,
    )
    if out_metadata is not None:
        out_metadata.update(search_meta)
    return candidates


def analyze_experiment_history(
    *,
    families: list[str],
    base_dir: str = "experiments",
    baseline_by_family: dict[str, str | None] | None = None,
    history_limit_per_family: int | None = None,
) -> dict[str, Any]:
    baseline_by_family = {**DEFAULT_BASELINES, **(baseline_by_family or {})}
    memory = load_research_memory(base_dir)
    scorecards = build_family_scorecards(families=families, base_dir=base_dir)
    scorecard_records = scorecards_to_records(scorecards)
    summary: dict[str, Any] = {"families": {}}
    for family in families:
        history = _load_detailed_history(family, base_dir, limit=history_limit_per_family)
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
        # Template counts need the full index (not the window-limited history) so
        # structural-template handoff thresholds are based on total evidence, not recency.
        _full_index_for_tc = load_prior_results(family=family, base_dir=base_dir)
        template_counts = _template_counts(_full_index_for_tc)
        stagnation = _stagnation_batches(history, lookback=10)
        summary["families"][family] = {
            "baseline_name": baseline_by_family.get(family),
            "history_count": int(len(history)),
            "scorecard": scorecard_records.get(family, {}),
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
        scorecard = family_analysis.get("scorecard") or {}
        score = base_weights.get(family, 0.05)
        score += min(max(viable_rate, 0.0), 1.0) * 0.20
        score += max(min(objective_max, 3.0), 0.0) * 0.04
        score += max(min(objective_mean, 2.0), -2.0) * 0.02
        score += _safe_float(scorecard.get("search_priority"), 0.0) * _safe_float(scorecard.get("confidence"), 0.0) * 0.25

        if family in {"ml_ranker", "rl_bandit"} and history_count >= 5 and viable_rate <= 0.0:
            score *= 0.45
        if family == "momentum" and viable_rate > 0.05:
            score *= 1.15
        if family == "superstock" and viable_rate > 0.10:
            score *= 1.05
        if _safe_float(scorecard.get("dead_zone_density"), 0.0) >= 0.75:
            score *= 0.85
        if _safe_float(scorecard.get("duplicate_saturation"), 0.0) >= 0.50:
            score *= 0.90
        if scorecard.get("recovery_signal"):
            score *= 1.10

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
            "scorecard": {
                "confidence": scorecard.get("confidence"),
                "evidence_weight": scorecard.get("evidence_weight"),
                "search_priority": scorecard.get("search_priority"),
                "exploration_budget_recommendation": scorecard.get("exploration_budget_recommendation"),
                "exploitation_budget_recommendation": scorecard.get("exploitation_budget_recommendation"),
                "stagnation_experiments": scorecard.get("stagnation_experiments"),
                "recovery_signal": scorecard.get("recovery_signal"),
            },
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
    if request.confirmation_required or bool(getattr(request, "holdout_check_required", False)):
        if request.max_experiments <= 0:
            return 0
        return 1
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
    new_idea_count = 0
    structurally_novel_count = 0
    uncommon_idea_count = 0
    repeat_branch_count = 0

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
        new_idea_count += sum(1 for item in metadata if item.get("is_new_idea"))
        structurally_novel_count += sum(1 for item in metadata if item.get("is_structurally_novel"))
        uncommon_idea_count += sum(1 for item in metadata if item.get("is_uncommon_idea"))
        repeat_branch_count += sum(1 for item in metadata if item.get("is_branch_repeat") or item.get("repeat_branch_flag"))
        family_reports[family] = {
            "budget": budget,
            "selected": len(configs),
            "fill_rate": (len(configs) / budget) if budget else 0.0,
            "search_space_size_estimate": _estimate_family_search_space_size(family),
            "explored_count": len(family_analysis.get("explored_hashes") or []),
            "source_types": family_source_types,
            "strategy_types": family_strategy_types,
            "idea_kinds": sorted({str(item.get("idea_kind")) for item in metadata if item.get("idea_kind")}),
            "new_idea_budget_buckets": sorted({str(item.get("new_idea_budget_bucket")) for item in metadata if item.get("new_idea_budget_bucket")}),
            "structurally_novel_count": sum(1 for item in metadata if item.get("is_structurally_novel")),
            "uncommon_idea_count": sum(1 for item in metadata if item.get("is_uncommon_idea")),
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
    confirmation_batch_small_valid = bool(request.confirmation_required and candidate_count > 0)
    holdout_check_small_valid = bool(getattr(request, "holdout_check_required", False) and candidate_count > 0)
    small_batch_valid = confirmation_batch_small_valid or holdout_check_small_valid
    pass_minimum = small_batch_valid or candidate_count >= min_viable or exhausted
    large_search_underfilled = (
        request.max_experiments >= request.large_search_threshold
        and candidate_count < min_viable
        and not exhausted
        and not confirmation_batch_small_valid
    )
    confirmation_underfilled = bool(request.confirmation_required) and candidate_count <= 0
    holdout_underfilled = bool(getattr(request, "holdout_check_required", False)) and candidate_count <= 0
    quality_flags: list[str] = []
    if candidate_count < min_viable and not small_batch_valid:
        quality_flags.append("too_small")
    if confirmation_batch_small_valid:
        quality_flags.append("confirmation_batch_small_valid")
    if holdout_check_small_valid:
        quality_flags.append("holdout_check_small_valid")
    if confirmation_underfilled:
        quality_flags.append("confirmation_batch_underfilled")
    if large_search_underfilled:
        quality_flags.append("large_search_underfilled")
    if holdout_underfilled:
        quality_flags.append("holdout_check_underfilled")
    if candidate_count > 0 and len(source_types) <= 1 and len(strategy_types) <= 1 and len(request.strategy_families) > 1:
        quality_flags.append("too_narrow")
    if request.new_idea_quota and new_idea_count < int(request.new_idea_quota) and not request.confirmation_required:
        quality_flags.append("new_idea_quota_underfilled")
    if request.uncommon_idea_quota and uncommon_idea_count < int(request.uncommon_idea_quota) and not request.confirmation_required:
        quality_flags.append("uncommon_idea_quota_underfilled")
    if request.repeat_branch_cap and repeat_branch_count > int(request.repeat_branch_cap) * max(1, len(request.strategy_families)):
        quality_flags.append("repeat_branch_cap_exceeded")
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
        "confirmation_batch_small_valid": confirmation_batch_small_valid,
        "holdout_check_small_valid": holdout_check_small_valid,
        "large_search_underfilled": large_search_underfilled,
        "confirmation_batch_underfilled": confirmation_underfilled,
        "holdout_check_underfilled": holdout_underfilled,
        "quality_flags": sorted(set(quality_flags)),
        "families_present": families_present,
        "family_reports": family_reports,
        "shortfall_reasons": sorted(shortfall_reasons),
        "duplicate_risk_count": duplicate_risk_count,
        "dead_zone_heavy_count": dead_zone_heavy_count,
        "local_refinement_count": local_refinement_count,
        "exploration_count": exploration_count,
        "new_idea_count": new_idea_count,
        "structurally_novel_count": structurally_novel_count,
        "uncommon_idea_count": uncommon_idea_count,
        "repeat_branch_count": repeat_branch_count,
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
    idea_id: str | None = None,
    idea_source: str | None = None,
    idea_kind: str | None = None,
    template_tags: list[str] | None = None,
    template_counts: dict[str, int] | None = None,
    lineage_root_config_hash: str | None = None,
    branch_budget: int | None = None,
    branch_budget_share: float | None = None,
    branch_budget_stance: str | None = None,
    branch_budget_reason: str | None = None,
    confirmation_state: str | None = None,
    confirmation_required: bool = False,
    confirmation_reason: str | None = None,
    confirmation_batch_id: str | None = None,
    confirmation_trial_kind: str | None = None,
    targeted_follow_up_required: bool = False,
    targeted_follow_up_reason: str | None = None,
    targeted_follow_up_type: str | None = None,
    targeted_follow_up_priority: float | None = None,
    targeted_follow_up_batch_id: str | None = None,
    holdout_check_required: bool = False,
    holdout_check_type: str | None = None,
    holdout_check_status: str | None = None,
    holdout_check_outcome: str | None = None,
    holdout_check_scope: str | None = None,
    holdout_check_batch_id: str | None = None,
    holdout_horizon_tags: list[str] | None = None,
    holdout_regime_tags: list[str] | None = None,
    structural_novelty_threshold: float = 0.55,
    uncommon_template_bonus: float = 0.10,
    synthesized_template_family: str | None = None,
    synthesis_rationale: str | None = None,
) -> dict[str, Any]:
    from experiment_store import compute_config_hash

    normalized = normalize_experiment_config(family, candidate)
    config_hash = compute_config_hash(family, normalized)
    idea_source_value = str(idea_source or source_type or "").strip() or "unknown"
    if source_type in {"confirmation_reproduction", "confirmation_check", "holdout_check"}:
        idea_kind = "holdout_confirmation"
    elif source_type in {"local_refinement"}:
        idea_kind = "branch_refinement"
    elif source_type in {"structural_synthesis"}:
        idea_kind = idea_kind or "new_strategy_logic_concept"
    elif source_type in {"idea_seed"}:
        idea_kind = "helper_idea"
    elif source_type in {"cross_family_hybrid"}:
        idea_kind = "cross_family_hybrid"
    elif source_type in {"model_based"}:
        idea_kind = "ml_architecture_probe"
    elif source_type in {"policy_learning"}:
        idea_kind = "rl_policy_probe"
    elif source_type in {"template_expansion"} and (template_tags or []):
        idea_kind = "template_variation"
    elif source_type in {"broad_exploration", "saturation_escape", "external_seed"}:
        idea_kind = "new_idea_probe"
    else:
        idea_kind = "exploration_variant"
    is_branch_repeat = source_type in {"local_refinement", "confirmation_reproduction", "confirmation_check", "holdout_check"} or bool(branch_budget)
    novelty_profile = _structural_novelty_profile(
        family=family,
        source_type=source_type,
        strategy_type=strategy_type,
        template_id=template_id,
        template_tags=template_tags,
        template_counts=template_counts,
        is_branch_repeat=is_branch_repeat,
        structural_novelty_threshold=structural_novelty_threshold,
        uncommon_template_bonus=uncommon_template_bonus,
    )
    is_new_idea = bool(novelty_profile["is_structurally_novel"])
    novelty_reason = (
        branch_budget_reason
        or reason_selected
        or (
            "branch repeat / local refinement"
            if is_branch_repeat
            else novelty_profile["structural_novelty_reason"]
        )
    )
    budget_bucket = (
        "confirmation"
        if source_type in {"confirmation_reproduction", "confirmation_check", "holdout_check"}
        else "uncommon_idea"
        if novelty_profile["is_uncommon_idea"]
        else "new_idea"
        if is_new_idea
        else "branch_refinement"
        if is_branch_repeat
        else "parameter_exploration"
    )
    return {
        "config": normalized,
        "config_hash": config_hash,
        "strategy_type": strategy_type,
        "source_type": source_type,
        "idea_source": idea_source_value,
        "idea_kind": idea_kind,
        "template_id": template_id,
        "hypothesis": hypothesis,
        "reason_selected": reason_selected,
        "novelty_reason": novelty_reason,
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
        "idea_id": idea_id,
        "is_new_idea": is_new_idea,
        "is_structurally_novel": bool(novelty_profile["is_structurally_novel"]),
        "is_out_of_box": bool(novelty_profile["is_out_of_box"]),
        "is_uncommon_idea": bool(novelty_profile["is_uncommon_idea"]),
        "structural_novelty_score": novelty_profile["structural_novelty_score"],
        "structural_novelty_reason": novelty_profile["structural_novelty_reason"],
        "is_branch_repeat": is_branch_repeat,
        "repeat_branch_flag": is_branch_repeat,
        "repeat_branch_depth": 0,
        "new_idea_budget_bucket": budget_bucket,
        "uncommon_idea_reason": novelty_profile["uncommon_idea_reason"],
        "template_tags": template_tags or [],
        "lineage_root_config_hash": lineage_root_config_hash,
        "branch_budget": branch_budget,
        "branch_budget_share": branch_budget_share,
        "branch_budget_stance": branch_budget_stance,
        "branch_budget_reason": branch_budget_reason,
        "synthesized_template_family": synthesized_template_family,
        "synthesis_rationale": synthesis_rationale,
        "confirmation_state": confirmation_state,
        "confirmation_required": confirmation_required,
        "confirmation_reason": confirmation_reason,
        "confirmation_batch_id": confirmation_batch_id,
        "confirmation_trial_kind": confirmation_trial_kind,
        "targeted_follow_up_required": targeted_follow_up_required,
        "targeted_follow_up_reason": targeted_follow_up_reason,
        "targeted_follow_up_type": targeted_follow_up_type,
        "targeted_follow_up_priority": targeted_follow_up_priority,
        "targeted_follow_up_batch_id": targeted_follow_up_batch_id,
        "holdout_check_required": holdout_check_required,
        "holdout_check_type": holdout_check_type,
        "holdout_check_status": holdout_check_status,
        "holdout_check_outcome": holdout_check_outcome,
        "holdout_check_scope": holdout_check_scope,
        "holdout_check_batch_id": holdout_check_batch_id,
        "holdout_horizon_tags": holdout_horizon_tags or [],
        "holdout_regime_tags": holdout_regime_tags or [],
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


# Per-synth-class override variants for the fd-probe floor.
# Each list entry is a dict of overrides on top of the synth-class default config.
# The floor tries variants in order until one passes deterministic safety checks.
# Only valid choices from the momentum/superstock/etc. search space are listed.
_FD_PROBE_VARIANT_OVERRIDES: dict[str, list[dict[str, Any]]] = {
    "ranking_rethink": [
        {},
        {"REBAL_WEEKS": 2, "MIN_HOLD_DAYS": 15},
        {"MA_WEEKS": 24, "FG_MIN": 22.0},
        {"REBAL_WEEKS": 2, "MIN_HOLD_DAYS": 10, "SKIP_WEEKS": 2},
        {"LOOKBACK_WEEKS": 26, "TOP_PCT": 0.015, "EXIT_PCT_RANK": 0.95},
    ],
    "regime_filter": [
        {},
        {"MA_WEEKS": 24, "STOP_TYPE": "fixed", "STOP_LOSS_PCT": 0.15},
        {"TOP_PCT": 0.015, "REBAL_WEEKS": 2, "MIN_HOLD_DAYS": 10},
        {"LOOKBACK_WEEKS": 39, "FG_MIN": 22.0, "MA_WEEKS": 24},
        {"SKIP_WEEKS": 3, "FG_MIN": 30.0, "LOOKBACK_WEEKS": 39},
    ],
    "breadth_filter": [
        {},
        {"MA_WEEKS": 24, "REBAL_WEEKS": 2},
        {"TOP_PCT": 0.015, "FG_MIN": 30.0},
    ],
    "state_model": [
        {},
        {"MA_WEEKS": 24, "REBAL_WEEKS": 2},
        {"TOP_PCT": 0.015, "EXIT_PCT_RANK": 0.95, "RANK_EXIT_CONFIRM": 1},
    ],
    "cross_family_mechanism": [
        {},
        {"REBAL_WEEKS": 2, "MA_WEEKS": 24},
        {"LOOKBACK_WEEKS": 26, "FG_MIN": 22.0},
    ],
    "portfolio_overlay": [
        {},
        {"STOP_LOSS_PCT": 0.15, "MA_WEEKS": 24},
        {"TOP_PCT": 0.015, "REBAL_WEEKS": 2},
    ],
}


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
        request = replace(request, source_idea_ids=request_source_idea_ids)
    history_limit_per_family = getattr(request, "history_limit_per_family", None)
    if history_limit_per_family is None:
        history_limit_per_family = max(8, min(16, int(request.max_experiments)))
        if request.confirmation_required or bool(getattr(request, "holdout_check_required", False)) or bool(
            getattr(request, "targeted_follow_up_required", False)
        ):
            history_limit_per_family = min(history_limit_per_family, 8)
        elif int(request.max_experiments) >= request.large_search_threshold:
            history_limit_per_family = min(history_limit_per_family, 12)
    analysis = analyze_experiment_history(
        families=request.strategy_families,
        base_dir=base_dir,
        baseline_by_family=baseline_by_family,
        history_limit_per_family=history_limit_per_family,
    )
    all_source_batch_ids = sorted(
        {
            str(batch_id)
            for family in request.strategy_families
            for batch_id in analysis["families"][family]["history_frame"].get("batch_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        }
    )
    if not request.source_batch_ids:
        request = replace(request, source_batch_ids=all_source_batch_ids)
    budgets, budget_negotiation = negotiate_family_budgets(request, analysis, guidance=analysis_guidance)
    scorecard_records = {
        family: analysis["families"].get(family, {}).get("scorecard", {})
        for family in request.strategy_families
    }
    candidate_configs: dict[str, list[dict[str, Any]]] = {}
    candidate_metadata: dict[str, list[dict[str, Any]]] = {}
    confirmation_mode = bool(request.confirmation_required)
    targeted_follow_up_mode = bool(getattr(request, "targeted_follow_up_required", False))
    targeted_follow_up_type = getattr(request, "targeted_follow_up_type", None)
    targeted_follow_up_reason = getattr(request, "targeted_follow_up_reason", None)
    targeted_follow_up_priority = float(getattr(request, "targeted_follow_up_priority", 0.0) or 0.0)
    targeted_follow_up_batch_id = getattr(request, "targeted_follow_up_batch_id", None)
    holdout_check_required = bool(getattr(request, "holdout_check_required", False))
    holdout_check_type = getattr(request, "holdout_check_type", None)
    holdout_check_status = getattr(request, "holdout_check_status", None)
    holdout_check_outcome = getattr(request, "holdout_check_outcome", None)
    holdout_check_scope = getattr(request, "holdout_check_scope", None)
    holdout_check_batch_id = getattr(request, "holdout_check_batch_id", None)
    holdout_horizon_tags = list(getattr(request, "holdout_horizon_tags", None) or [])
    holdout_regime_tags = list(getattr(request, "holdout_regime_tags", None) or [])
    confirmation_focus_family = request.confirmation_focus_family
    if not confirmation_focus_family and len(request.strategy_families) == 1:
        confirmation_focus_family = request.strategy_families[0]
    official_index = load_results_index(base_dir)
    official_hashes_by_family: dict[str, set[str]] = {}
    if not official_index.empty and {"strategy_family", "config_hash"}.issubset(official_index.columns):
        hash_frame = official_index[["strategy_family", "config_hash"]].dropna()
        for hash_family, hash_group in hash_frame.groupby(hash_frame["strategy_family"].astype(str)):
            official_hashes_by_family[str(hash_family)] = set(hash_group["config_hash"].astype(str))
    lineage_summary = build_lineage_summary(
        official_index,
        persisted_records=load_lineage_state_records(base_dir),
        latest_batch=None,
        include_histories=False,
        include_records=False,
    )
    lineage_cycle_mode = (
        "confirmation"
        if confirmation_mode or targeted_follow_up_mode or holdout_check_required
        else "large-search"
        if request.max_experiments >= request.large_search_threshold
        else "normal_exploration"
    )
    branch_budgets = dict(request.branch_budgets or {})
    branch_budget_rationale = dict(request.branch_budget_rationale or {})
    if not branch_budgets:
        branch_budgets, branch_budget_rationale = build_branch_budget_plan(
            lineage_summary,
            budgets,
            cycle_mode=lineage_cycle_mode,
            confirmation_family=confirmation_focus_family,
            confirmation_required=confirmation_mode,
            targeted_follow_up_required=targeted_follow_up_mode,
            holdout_check_required=holdout_check_required,
        )
    repeat_branch_cap = max(1, int(request.repeat_branch_cap or max(1, int(round(request.max_experiments * 0.25)))))

    def _branch_seed_row(family_history: pd.DataFrame, branch_entry: dict[str, Any]) -> pd.Series | None:
        branch_hash = str(
            branch_entry.get("best_branch_config_hash")
            or branch_entry.get("branch_root_config_hash")
            or ""
        ).strip()
        if not branch_hash or family_history.empty:
            return None
        match = family_history[family_history["config_hash"].astype(str) == branch_hash]
        if match.empty:
            return None
        return match.iloc[0]

    reasoning: dict[str, Any] = {
        "source_batch_ids": request.source_batch_ids or all_source_batch_ids,
        "source_idea_ids": request.source_idea_ids or [],
        "family_budget_decision": budgets,
        "family_budget_negotiation": budget_negotiation,
        "branch_budgets": branch_budgets,
        "branch_budget_rationale": branch_budget_rationale,
        "family_scorecards": scorecard_records,
        "analysis_guidance": analysis_guidance.get("next_focus") if analysis_guidance else [],
        "history_limit_per_family": history_limit_per_family,
        "novelty_policy": {
            "exploration_fraction": request.exploration_fraction,
            "exploitation_fraction": request.exploitation_fraction,
            "novelty_floor": request.novelty_floor,
            "novelty_floor_mode": "per_candidate_adaptive",
            "template_fraction": request.template_fraction,
            "cross_family_fraction": request.cross_family_fraction,
            "max_near_duplicate_distance": request.max_near_duplicate_distance,
            "stagnation_escape_batches": request.stagnation_escape_batches,
            "allow_external_seeds": request.allow_external_seeds,
            "new_idea_budget": request.new_idea_budget,
            "refinement_budget": request.refinement_budget,
            "confirmation_budget": request.confirmation_budget,
            "new_idea_quota": request.new_idea_quota,
            "uncommon_idea_quota": request.uncommon_idea_quota,
            "structural_idea_budget": request.structural_idea_budget,
            "out_of_box_budget": request.out_of_box_budget,
            "max_routine_template_budget": request.max_routine_template_budget,
            "repeat_branch_cap": request.repeat_branch_cap,
            "max_same_template_per_cycle": request.max_same_template_per_cycle,
            "max_same_lineage_per_cycle": request.max_same_lineage_per_cycle,
            "structural_novelty_threshold": request.structural_novelty_threshold,
            "uncommon_template_bonus": request.uncommon_template_bonus,
        },
        "confirmation": {
            "state": request.confirmation_state,
            "required": bool(request.confirmation_required),
            "reason": request.confirmation_reason,
            "batch_id": request.confirmation_batch_id,
            "focus_family": request.confirmation_focus_family,
            "batch_experiments": request.confirmation_batch_experiments,
        },
        "targeted_follow_up": {
            "required": targeted_follow_up_mode,
            "reason": targeted_follow_up_reason,
            "type": targeted_follow_up_type,
            "priority": targeted_follow_up_priority,
            "batch_id": targeted_follow_up_batch_id,
        },
        "holdout_check": {
            "required": holdout_check_required,
            "type": holdout_check_type,
            "status": holdout_check_status,
            "outcome": holdout_check_outcome,
            "scope": holdout_check_scope,
            "batch_id": holdout_check_batch_id,
            "horizon_tags": holdout_horizon_tags,
            "regime_tags": holdout_regime_tags,
        },
        "holdout_check_required": holdout_check_required,
        "holdout_check_type": holdout_check_type,
        "holdout_check_status": holdout_check_status,
        "holdout_check_outcome": holdout_check_outcome,
        "holdout_check_scope": holdout_check_scope,
        "holdout_check_batch_id": holdout_check_batch_id,
        "holdout_horizon_tags": holdout_horizon_tags,
        "holdout_regime_tags": holdout_regime_tags,
        "families": {},
    }
    memory = load_research_memory(base_dir)
    proposal_seed_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(request.proposal_id)) % 100_000
    reasoning["novelty_policy"]["proposal_seed_offset"] = proposal_seed_offset

    cycle_new_idea_budget = max(
        1,
        int(
            request.new_idea_budget
            if request.new_idea_budget is not None
            else max(1, int(round(request.max_experiments * 0.20)))
        ),
    )
    cycle_new_idea_floor = max(1, int(round(cycle_new_idea_budget / max(1, len(request.strategy_families)))))
    cycle_uncommon_idea_budget = max(
        1,
        int(
            request.uncommon_idea_quota
            if request.uncommon_idea_quota is not None
            else max(1, int(round(cycle_new_idea_budget * 0.35)))
        ),
    )
    cycle_uncommon_idea_floor = max(1, int(round(cycle_uncommon_idea_budget / max(1, len(request.strategy_families)))))
    # Load the latest persisted idea-yield summary for per-family floor adjustments.
    # Falls back to empty dict gracefully when no report exists yet.
    _idea_yield_cache = load_latest_idea_yield_summary(base_dir) or {}

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
        family_confirmation_mode = confirmation_mode and family == confirmation_focus_family
        confirmation_trial_budget = max(0, budget)
        confirmation_focus_only = family_confirmation_mode
        if family_confirmation_mode:
            explicit_confirmation_budget = int(request.confirmation_budget or request.confirmation_batch_experiments or 0)
            if explicit_confirmation_budget > 0:
                confirmation_trial_budget = max(1, min(budget, explicit_confirmation_budget))
            else:
                confirmation_trial_budget = max(1, min(budget, max(4, int(round(max(1, budget) * 0.25)))))
        new_idea_floor = 0 if family_confirmation_mode and budget <= 4 else max(1, int(round(max(1, budget) * 0.25)))
        new_idea_floor = max(new_idea_floor, min(budget, cycle_new_idea_floor))
        if request.new_idea_quota is not None and request.new_idea_quota > 0:
            per_family_quota_floor = max(1, int(round(float(request.new_idea_quota) / max(1, len(request.strategy_families)))))
            new_idea_floor = max(new_idea_floor, min(budget, per_family_quota_floor))
        # --- Idea-yield feedback: adjust new_idea_floor based on historical yield quality ---
        _fam_yield = (_idea_yield_cache.get("families") or {}).get(family, {})
        _fam_idea_state = str(_fam_yield.get("idea_state") or "untested").strip().lower()
        _fam_idea_quality = float(_fam_yield.get("idea_quality_score") or 0.0)
        _fam_retired_share = float(_fam_yield.get("idea_retired_share") or 0.0)
        _fam_promising_share = float(_fam_yield.get("idea_promising_share") or 0.0)
        _fam_structural_state = str(_fam_yield.get("structural_family_state") or _fam_idea_state).strip().lower()
        _fam_structural_state_previous = str(_fam_yield.get("structural_family_state_previous") or "").strip() or None
        _fam_structural_promotion_state = str(_fam_yield.get("structural_family_promotion_state") or "floor_protected").strip().lower()
        _fam_structural_transition_reason = str(_fam_yield.get("structural_family_transition_reason") or "").strip() or None
        _yield_floor_action = "default"
        if not family_confirmation_mode and budget > 1:
            if _fam_idea_state == "retired" or _fam_retired_share >= 0.70:
                # Nearly all idea kinds exhausted — shrink floor, save budget for refinement
                new_idea_floor = min(new_idea_floor, max(1, int(round(cycle_new_idea_floor * 0.50))))
                _yield_floor_action = "reduced_retired"
            elif _fam_idea_state == "stale" or (_fam_retired_share >= 0.45 and _fam_idea_quality < 0.30):
                # Portfolio is stale — moderate floor reduction
                new_idea_floor = min(new_idea_floor, max(1, int(round(cycle_new_idea_floor * 0.75))))
                _yield_floor_action = "reduced_stale"
            elif _fam_idea_state == "promising" and _fam_idea_quality >= 0.55:
                # High-yield ideas available — keep floor high to sustain momentum
                new_idea_floor = max(new_idea_floor, cycle_new_idea_floor)
                _yield_floor_action = "maintained_promising"
            if _fam_structural_promotion_state == "graduated_structural_family":
                new_idea_floor = max(new_idea_floor, min(budget, cycle_new_idea_floor + 1))
                _yield_floor_action = "graduated_structural_family"
            elif _fam_structural_promotion_state == "yield_supported" and _yield_floor_action == "default":
                new_idea_floor = max(new_idea_floor, cycle_new_idea_floor)
                _yield_floor_action = "yield_supported"
            elif _fam_structural_promotion_state == "floor_protected" and _fam_structural_state == "untested" and _yield_floor_action == "default":
                new_idea_floor = max(new_idea_floor, cycle_new_idea_floor)
                _yield_floor_action = "floor_protected"
            elif _fam_structural_promotion_state in {"decaying", "retired"} and _yield_floor_action == "default":
                new_idea_floor = min(new_idea_floor, max(1, int(round(cycle_new_idea_floor * (0.75 if _fam_structural_promotion_state == "decaying" else 0.50)))))
                _yield_floor_action = _fam_structural_promotion_state
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
        if budget > new_idea_floor:
            exploit_n = min(exploit_n, max(1, budget - new_idea_floor))
        explore_n = max(0, budget - exploit_n)
        _branch_budget_total = sum(int(e.get("branch_budget") or 0) for e in (
            [dict(en) for en in (branch_budgets.get(family) or []) if int((en or {}).get("branch_budget") or 0) > 0]
        ))
        _branch_dominance = _branch_budget_total / max(1, budget)
        _eff_template_frac, _eff_cf_frac = _adaptive_budget_fractions(
            cycle_mode=lineage_cycle_mode,
            stagnation_batches=stagnation_batches,
            viable_rate=float(family_analysis.get("viable_rate") or 0.0),
            base_template_fraction=request.template_fraction,
            base_cross_family_fraction=request.cross_family_fraction,
            branch_dominance=_branch_dominance,
            has_multiple_families=len(request.strategy_families) > 1,
        )
        template_n = max(1, int(round(explore_n * _eff_template_frac))) if explore_n else 0
        cross_family_n = 0
        if len(request.strategy_families) > 1 and explore_n:
            cross_family_n = max(1, int(round(explore_n * _eff_cf_frac)))
            cross_family_n = min(cross_family_n, explore_n)
        local_explore_n = max(0, explore_n - template_n - cross_family_n)
        if family_confirmation_mode:
            exploit_n = max(1, min(budget, max(1, min(confirmation_trial_budget, int(round(max(1, budget) * 0.25))))))
            explore_n = max(0, budget - exploit_n)
            template_n = max(1, int(round(explore_n * request.template_fraction))) if explore_n else 0
            cross_family_n = max(1, int(round(explore_n * request.cross_family_fraction))) if len(request.strategy_families) > 1 and explore_n else 0
            cross_family_n = min(cross_family_n, explore_n)
            local_explore_n = max(0, explore_n - template_n - cross_family_n)

        top = select_top_configs(history, family, limit=max(1, min(8, exploit_n or 1)))
        seen_hashes = set(explored_hashes)
        if family_confirmation_mode or targeted_follow_up_mode or holdout_check_required:
            # Confirmation may intentionally rerun the exact seed, but local neighborhoods
            # should not waste execution slots on older already-seen configs outside the hot window.
            seen_hashes.update(official_hashes_by_family.get(family, set()))
        seen_signatures = set(explored_signatures.values())
        selection_pool: list[dict[str, Any]] = []
        branch_repeat_counts: dict[str, int] = {}
        template_cycle_counts: dict[str, int] = {}
        lineage_cycle_counts: dict[str, int] = {}
        branch_repeat_total = 0
        # Structural-template balancing: guarantee fair early exploration across all 4
        # protected momentum templates.  Templates below _STRUCTURAL_EVIDENCE_THRESHOLD
        # receive guaranteed floor slots; templates above it are capped to
        # _STRUCTURAL_REPEAT_CAP per cycle while any others remain below threshold.
        _STRUCTURAL_MOMENTUM_TEMPLATES: frozenset[str] = frozenset({
            "momentum_vol_scaling_v1",
            "momentum_vol_scaling_downside_v1",
            "momentum_drift_regime_v1",
            "momentum_drift_regime_strict_v1",
        })
        _STRUCTURAL_EVIDENCE_THRESHOLD = 8  # experiments needed before yield-based favoritism
        _STRUCTURAL_REPEAT_CAP = 2          # max per-cycle slots for "ahead" template when others behind
        _structural_tc = (family_analysis.get("template_counts") or {}) if family == "momentum" else {}
        _structural_behind: frozenset[str] = frozenset(
            t for t in _STRUCTURAL_MOMENTUM_TEMPLATES
            if int(_structural_tc.get(t, 0)) < _STRUCTURAL_EVIDENCE_THRESHOLD
        ) if family == "momentum" else frozenset()
        _structural_ahead: frozenset[str] = frozenset(
            t for t in _STRUCTURAL_MOMENTUM_TEMPLATES
            if int(_structural_tc.get(t, 0)) >= _STRUCTURAL_EVIDENCE_THRESHOLD
        ) if family == "momentum" else frozenset()
        _has_structural_behind = bool(_structural_behind)

        # ── Yield-driven handoff state ──────────────────────────────────────
        # Three states:
        #   floor_protected      – all templates below evidence threshold
        #   mixed_floor_and_yield – some graduated, some still behind
        #   yield_driven          – all templates above threshold
        if not _structural_ahead:
            _handoff_state = "floor_protected"
        elif _structural_behind:
            _handoff_state = "mixed_floor_and_yield"
        else:
            _handoff_state = "yield_driven"

        # Per-template yield scores (only meaningful for graduated templates).
        # Score = viable_rate * 0.5 + beat_rate * 0.3 + clamped_mean_score * 0.2
        # All inputs clipped to [0, 1]; minimum floor of 0.01 avoids zero allocation.
        _structural_yield_scores: dict[str, float] = {}
        if family == "momentum" and _structural_ahead and not history.empty and "template_id" in history.columns:
            _tid_col = history["template_id"].astype(str)
            for _ytid in _structural_ahead:
                _ysub = history[_tid_col == _ytid]
                if _ysub.empty:
                    _structural_yield_scores[_ytid] = 0.01
                    continue
                _viable_rate = float(
                    _ysub["viable"].map(lambda v: bool(v) if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}).mean()
                ) if "viable" in _ysub.columns else 0.0
                _beat_col = _ysub.get("beats_baseline_objective") if hasattr(_ysub, "get") else None
                _beat_rate = float(
                    _ysub["beats_baseline_objective"].map(lambda v: bool(v) if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}).mean()
                ) if "beats_baseline_objective" in _ysub.columns else 0.0
                _mean_score = float(
                    pd.to_numeric(_ysub["objective_score"], errors="coerce").fillna(0.0).mean()
                ) if "objective_score" in _ysub.columns else 0.0
                _structural_yield_scores[_ytid] = max(
                    0.01,
                    min(1.0, _viable_rate) * 0.5
                    + min(1.0, _beat_rate) * 0.3
                    + min(1.0, max(0.0, _mean_score)) * 0.2,
                )

        # Per-template slot cap used in _cycle_caps_exhausted.
        # mixed_floor_and_yield → flat REPEAT_CAP for ahead templates (anti-monopoly)
        # yield_driven           → proportional to yield score, bounded by monopoly cap
        _structural_budget_slots = max(len(_STRUCTURAL_MOMENTUM_TEMPLATES), int(round(budget * 0.33)))
        _structural_monopoly_limit = max(1, int(_structural_budget_slots // 2))
        _structural_per_template_cap: dict[str, int] = {}
        if _handoff_state == "mixed_floor_and_yield":
            _structural_per_template_cap = {tid: _STRUCTURAL_REPEAT_CAP for tid in _structural_ahead}
            _handoff_rationale = (
                f"handoff_state=mixed_floor_and_yield; "
                f"{len(_structural_behind)} behind / {len(_structural_ahead)} ahead; "
                f"ahead templates capped at repeat_cap={_STRUCTURAL_REPEAT_CAP}"
            )
        elif _handoff_state == "yield_driven":
            _total_yield = sum(_structural_yield_scores.values()) or 1.0
            for _ytid in _structural_ahead:
                _frac = _structural_yield_scores.get(_ytid, 0.01) / _total_yield
                _structural_per_template_cap[_ytid] = min(
                    _structural_monopoly_limit,
                    max(1, int(round(_frac * _structural_budget_slots))),
                )
            _handoff_rationale = (
                f"handoff_state=yield_driven; all {len(_STRUCTURAL_MOMENTUM_TEMPLATES)} templates graduated; "
                f"yield_scores={{{', '.join(f'{k}:{v:.3f}' for k, v in sorted(_structural_yield_scores.items()))}}}; "
                f"caps={_structural_per_template_cap}"
            )
        else:
            _handoff_rationale = (
                f"handoff_state=floor_protected; "
                f"all {len(_structural_behind)} templates below threshold={_STRUCTURAL_EVIDENCE_THRESHOLD}"
            )

        max_same_template = max(
            1,
            int(
                request.max_same_template_per_cycle
                if request.max_same_template_per_cycle is not None
                else max(2, int(round(max(1, budget) * 0.35)))
            ),
        )
        max_same_lineage = max(
            1,
            int(
                request.max_same_lineage_per_cycle
                if request.max_same_lineage_per_cycle is not None
                else max(2, int(round(max(1, budget) * 0.35)))
            ),
        )
        family_ideas = [record for record in helper_ideas if record.get("family") in {None, family}]
        structural_family_ideas = [
            record
            for record in family_ideas
            if bool(record.get("is_structurally_novel"))
            or bool(record.get("is_out_of_box"))
            or bool(record.get("is_uncommon_idea"))
        ]
        # Family-discovery probe ideas get a guaranteed floor budget even in confirmation mode.
        fd_probe_ideas = [
            r for r in structural_family_ideas
            if r.get("idea_source") == "family_discovery"
        ]
        routine_family_ideas = [record for record in family_ideas if record not in structural_family_ideas]
        structural_yield_records = [
            record
            for record in ((_idea_yield_cache.get("by_synthesized_template_family") or {}).values())
            if str(record.get("family") or "") == family
        ]
        structural_feedback = dict(_fam_yield.get("structural_family_feedback") or {})
        structural_successful_families = {
            str(record.get("synthesized_template_family") or record.get("idea_kind") or "").strip().lower()
            for record in structural_yield_records
            if str(record.get("idea_state") or "").strip().lower() in {"promising", "active"}
            and float(record.get("idea_quality_score") or 0.0) >= 0.55
        }
        structural_idea_budget = max(
            1,
            int(
                request.structural_idea_budget
                if request.structural_idea_budget is not None
                else max(1, int(round(max(1, budget) * 0.30)))
            ),
        )
        structural_idea_budget = min(budget, structural_idea_budget + len(structural_successful_families))
        structural_execution_lane_enabled = bool(
            request.structural_execution_lane_enabled if request.structural_execution_lane_enabled is not None else True
        )
        structural_execution_lane_budget = max(
            0,
            int(
                request.structural_execution_lane_budget
                if request.structural_execution_lane_budget is not None
                else structural_idea_budget
            ),
        )
        structural_execution_lane_min_candidates = max(
            1,
            int(
                request.structural_execution_lane_min_candidates
                if request.structural_execution_lane_min_candidates is not None
                else max(1, int(round(max(1, budget) * 0.15)))
            ),
        )
        structural_execution_lane_repeat_cap = max(
            1,
            int(
                request.structural_execution_lane_repeat_cap
                if request.structural_execution_lane_repeat_cap is not None
                else repeat_branch_cap
            ),
        )
        if _fam_structural_promotion_state == "graduated_structural_family":
            structural_execution_lane_budget = max(structural_execution_lane_budget, cycle_new_idea_floor + 1)
            structural_execution_lane_repeat_cap = max(structural_execution_lane_repeat_cap, 3)
        elif _fam_structural_promotion_state == "yield_supported":
            structural_execution_lane_budget = max(structural_execution_lane_budget, cycle_new_idea_floor)
            structural_execution_lane_repeat_cap = max(structural_execution_lane_repeat_cap, 2)
        elif _fam_structural_promotion_state in {"decaying", "retired"}:
            structural_execution_lane_budget = min(structural_execution_lane_budget, max(1, cycle_new_idea_floor // 2 or 1))
            structural_execution_lane_repeat_cap = min(structural_execution_lane_repeat_cap, 1)
        structural_execution_lane_budget = min(
            budget,
            max(structural_execution_lane_min_candidates, structural_execution_lane_budget),
            max(1, int(round(max(1, budget) * 0.33))),
        )
        out_of_box_budget = max(
            1,
            int(
                request.out_of_box_budget
                if request.out_of_box_budget is not None
                else max(1, int(round(structural_idea_budget * 0.50)))
            ),
        )
        uncommon_idea_budget = max(
            1,
            int(
                request.uncommon_idea_budget
                if request.uncommon_idea_budget is not None
                else max(1, int(round(structural_idea_budget * 0.35)))
            ),
        )
        max_routine_template_budget = max(
            1,
            int(
                request.max_routine_template_budget
                if request.max_routine_template_budget is not None
                else max(1, budget - structural_idea_budget)
            ),
        )
        structural_family_targets: dict[str, int] = {}
        for idea in structural_family_ideas:
            synth_family = _structural_synthesis_family(idea)
            if synth_family in structural_successful_families:
                structural_family_targets[synth_family] = max(structural_family_targets.get(synth_family, 0), 2)
            else:
                structural_family_targets.setdefault(synth_family, 1)
        structural_budget_state = (
            "yield_driven"
            if _fam_structural_promotion_state == "graduated_structural_family" or structural_successful_families
            else "yield_supported"
            if _fam_structural_promotion_state == "yield_supported"
            else "decaying"
            if _fam_structural_promotion_state in {"decaying", "retired"}
            else "floor_protected"
            if structural_family_ideas
            else "no_structural_ideas"
        )
        family_branch_budget_entries = [
            dict(entry)
            for entry in (branch_budgets.get(family) or [])
            if int((entry or {}).get("branch_budget") or 0) > 0
        ]
        family_branch_budget_entries.sort(
            key=lambda item: (
                -int(item.get("branch_budget") or 0),
                -float(item.get("branch_trust_score") or item.get("branch_score") or 0.0),
                str(item.get("branch_root_config_hash") or ""),
            )
        )
        guidance_focus = None
        if analysis_guidance:
            for item in analysis_guidance.get("next_focus", []):
                if item.get("family") == family:
                    guidance_focus = item
                    break

        def _repeat_branch_key(metadata: dict[str, Any]) -> str:
            return str(
                metadata.get("lineage_root_config_hash")
                or metadata.get("parent_config_hash")
                or metadata.get("template_id")
                or metadata.get("region_label")
                or ""
            )

        def _lineage_cycle_key(metadata: dict[str, Any]) -> str:
            return str(metadata.get("lineage_root_config_hash") or metadata.get("parent_config_hash") or metadata.get("config_hash") or "")

        def _cycle_caps_exhausted(metadata: dict[str, Any]) -> bool:
            source_type = str(metadata.get("source_type") or "")
            protected = source_type in {"confirmation_reproduction", "confirmation_check", "holdout_check"}
            if protected and int(metadata.get("confirmation_slot_index") or 0) <= 1:
                return False
            template_key = _template_cycle_key(metadata.get("template_id"), source_type)
            if template_key and template_cycle_counts.get(template_key, 0) >= max_same_template and not metadata.get("is_uncommon_idea"):
                return True
            # Structural template per-cycle cap, graduated by handoff state:
            #   mixed_floor_and_yield → flat REPEAT_CAP (anti-monopoly while others behind)
            #   yield_driven           → yield-proportional cap (with monopoly ceiling)
            if (
                template_key in _structural_per_template_cap
                and template_cycle_counts.get(template_key, 0) >= _structural_per_template_cap[template_key]
            ):
                return True
            lineage_key = _lineage_cycle_key(metadata)
            if lineage_key and lineage_cycle_counts.get(lineage_key, 0) >= max_same_lineage and not metadata.get("is_structurally_novel"):
                return True
            return False

        def _record_cycle_caps(metadata: dict[str, Any]) -> None:
            source_type = str(metadata.get("source_type") or "")
            template_key = _template_cycle_key(metadata.get("template_id"), source_type)
            if template_key:
                template_cycle_counts[template_key] = template_cycle_counts.get(template_key, 0) + 1
            lineage_key = _lineage_cycle_key(metadata)
            if lineage_key:
                lineage_cycle_counts[lineage_key] = lineage_cycle_counts.get(lineage_key, 0) + 1

        def _repeat_branch_exhausted(metadata: dict[str, Any]) -> bool:
            if family_confirmation_mode and metadata.get("source_type") in {
                "confirmation_reproduction",
                "confirmation_check",
                "holdout_check",
            }:
                return False
            if not metadata.get("is_branch_repeat"):
                return False
            if branch_repeat_total >= max(0, budget - new_idea_floor):
                return True
            key = _repeat_branch_key(metadata)
            if not key:
                return False
            return branch_repeat_counts.get(key, 0) >= repeat_branch_cap

        def _record_repeat_branch(metadata: dict[str, Any]) -> None:
            nonlocal branch_repeat_total
            metadata["repeat_branch_depth"] = branch_repeat_counts.get(_repeat_branch_key(metadata), 0)
            if not metadata.get("is_branch_repeat"):
                return
            key = _repeat_branch_key(metadata)
            if not key:
                return
            branch_repeat_counts[key] = branch_repeat_counts.get(key, 0) + 1
            branch_repeat_total += 1

        def _append_confirmation_reproduction(seed_row: pd.Series) -> None:
            if not family_confirmation_mode or len(selection_pool) >= confirmation_trial_budget:
                return
            exact_config = dict(seed_row.get("config") or {})
            if not exact_config:
                return
            normalized = normalize_experiment_config(family, exact_config)
            config_hash = compute_config_hash(family, normalized)
            if any(item.get("config_hash") == config_hash for item in selection_pool):
                return
            novelty = SimpleNamespace(
                exact_duplicate=False,
                near_duplicate=False,
                near_duplicate_of=seed_row.get("config_hash"),
                duplicate_risk="confirmation_exact",
                novelty_score=1.0,
                selection_score=max(1.0, _safe_float(seed_row.get("objective_score"), 0.0) + 0.5),
                objective_proxy=_safe_float(seed_row.get("objective_score"), 0.0),
                dead_zone_risk=0.0,
                dead_zone_flags=[],
            )
            metadata = _candidate_metadata(
                family=family,
                candidate=normalized,
                novelty=novelty,
                strategy_type=_family_strategy_type(family),
                source_type="confirmation_reproduction",
                template_id=seed_row.get("template_id") or "confirmation_reproduce",
                hypothesis=seed_row.get("hypothesis") or family_analysis.get("baseline_name"),
                reason_selected="exact reproduction to confirm a promoted winner",
                exploration_mode="confirmation",
                proposal_role="confirm",
                region_label="confirmation_exact",
                parent_config_hash=seed_row.get("config_hash"),
                source_batch_ids=request.source_batch_ids or all_source_batch_ids,
                source_proposal_id=request.proposal_id,
                source_idea_ids=[],
                template_counts=family_analysis.get("template_counts", {}),
                confirmation_state=request.confirmation_state,
                confirmation_required=request.confirmation_required,
                confirmation_reason=request.confirmation_reason,
                confirmation_batch_id=request.confirmation_batch_id,
                confirmation_trial_kind="reproduce",
                targeted_follow_up_required=targeted_follow_up_mode,
                targeted_follow_up_reason=targeted_follow_up_reason,
                targeted_follow_up_type=targeted_follow_up_type,
                targeted_follow_up_priority=targeted_follow_up_priority,
                targeted_follow_up_batch_id=targeted_follow_up_batch_id,
                holdout_check_required=holdout_check_required,
                holdout_check_type=holdout_check_type,
                holdout_check_status=holdout_check_status,
                holdout_check_outcome=holdout_check_outcome,
                holdout_check_scope=holdout_check_scope,
                holdout_check_batch_id=holdout_check_batch_id,
                holdout_horizon_tags=holdout_horizon_tags,
                holdout_regime_tags=holdout_regime_tags,
                structural_novelty_threshold=request.structural_novelty_threshold,
                uncommon_template_bonus=request.uncommon_template_bonus,
            )
            metadata["confirmation_slot_index"] = len(selection_pool) + 1
            selection_pool.append(metadata)
            _record_cycle_caps(metadata)
            seen_hashes.add(config_hash)
            seen_signatures.add(coarse_signature_key(family, normalized))

        def _append_confirmation_neighborhood(seed_row: pd.Series, *, holdout: bool = False) -> None:
            if not (family_confirmation_mode or holdout_check_required):
                return
            remaining = confirmation_trial_budget - len(selection_pool)
            if remaining <= 0:
                return
            exact_config = dict(seed_row.get("config") or {})
            if not exact_config:
                return
            neighbor_limit = max(1, remaining - 1)
            if family_confirmation_mode:
                neighbor_limit = max(1, min(neighbor_limit, int(round(max(1, confirmation_trial_budget) * 0.25))))
            neighbors = build_local_neighbors(
                family,
                exact_config,
                limit=neighbor_limit,
                seed=request.seed + proposal_seed_offset + idx + len(selection_pool) + 17,
                dead_zones=dead_zones,
                explored_hashes=seen_hashes,
            )
            if not neighbors:
                return
            source_type = "confirmation_check" if not holdout else "holdout_check"
            region_label = "confirmation_local_neighborhood" if not holdout else "holdout_local_neighborhood"
            trial_kind = "neighborhood" if not holdout else "holdout_neighborhood"
            for neighbor in neighbors:
                if len(selection_pool) >= confirmation_trial_budget:
                    break
                _try_add_candidate(
                    neighbor,
                    source_type=source_type,
                    strategy_type=_family_strategy_type(family),
                    template_id=region_label,
                    hypothesis="Robustness check around the promoted winner.",
                    reason_selected="confirmation batch local neighborhood check" if not holdout else "holdout batch local neighborhood check",
                    exploration_mode="confirmation" if not holdout else "holdout",
                    proposal_role="confirm",
                    region_label=region_label,
                    parent_config_hash=seed_row.get("config_hash"),
                    source_idea_ids=[],
                    template_counts=family_analysis.get("template_counts", {}),
                    confirmation_state=request.confirmation_state,
                    confirmation_required=request.confirmation_required,
                    confirmation_reason=request.confirmation_reason,
                    confirmation_batch_id=request.confirmation_batch_id,
                    confirmation_trial_kind=trial_kind,
                    targeted_follow_up_required=targeted_follow_up_mode,
                    targeted_follow_up_reason=targeted_follow_up_reason,
                    targeted_follow_up_type=targeted_follow_up_type,
                    targeted_follow_up_priority=targeted_follow_up_priority,
                    targeted_follow_up_batch_id=targeted_follow_up_batch_id,
                    holdout_check_required=holdout_check_required,
                    holdout_check_type=holdout_check_type,
                    holdout_check_status=holdout_check_status,
                    holdout_check_outcome=holdout_check_outcome,
                    holdout_check_scope=holdout_check_scope,
                    holdout_check_batch_id=holdout_check_batch_id,
                    holdout_horizon_tags=holdout_horizon_tags,
                    holdout_regime_tags=holdout_regime_tags,
                    novelty_floor_override=0.0,
                    allow_near_duplicate=True,
                )

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
            idea_id: str | None = None,
            idea_source: str | None = None,
            idea_kind: str | None = None,
            synthesized_template_family: str | None = None,
            synthesis_rationale: str | None = None,
            structural_novelty_threshold: float = 0.55,
            uncommon_template_bonus: float = 0.10,
            template_tags: list[str] | None = None,
            template_counts: dict[str, int] | None = None,
            lineage_root_config_hash: str | None = None,
            branch_budget: int | None = None,
            branch_budget_share: float | None = None,
            branch_budget_stance: str | None = None,
            branch_budget_reason: str | None = None,
            confirmation_state: str | None = request.confirmation_state,
            confirmation_required: bool = bool(request.confirmation_required),
            confirmation_reason: str | None = request.confirmation_reason,
            confirmation_batch_id: str | None = request.confirmation_batch_id,
            confirmation_trial_kind: str | None = None,
            novelty_floor_override: float | None = None,
            allow_near_duplicate: bool = False,
            targeted_follow_up_required: bool = targeted_follow_up_mode,
            targeted_follow_up_reason: str | None = targeted_follow_up_reason,
            targeted_follow_up_type: str | None = targeted_follow_up_type,
            targeted_follow_up_priority: float | None = targeted_follow_up_priority,
            targeted_follow_up_batch_id: str | None = targeted_follow_up_batch_id,
            holdout_check_required: bool = holdout_check_required,
            holdout_check_type: str | None = holdout_check_type,
            holdout_check_status: str | None = holdout_check_status,
            holdout_check_outcome: str | None = holdout_check_outcome,
            holdout_check_scope: str | None = holdout_check_scope,
            holdout_check_batch_id: str | None = holdout_check_batch_id,
            holdout_horizon_tags: list[str] | None = None,
            holdout_regime_tags: list[str] | None = None,
        ) -> bool:
            effective_floor = (
                float(novelty_floor_override)
                if novelty_floor_override is not None
                else _get_novelty_floor(
                    exploration_mode,
                    stagnation_batches=stagnation_batches,
                    is_confirmation=family_confirmation_mode or bool(request.confirmation_required),
                    is_holdout=holdout_check_required,
                    base_floor=novelty_floor,
                )
            )
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
                novelty_floor=effective_floor,
                near_duplicate_distance=request.max_near_duplicate_distance,
            )
            if novelty is None:
                return False
            if novelty.near_duplicate and not allow_near_duplicate:
                return False
            normalized = normalize_experiment_config(family, candidate)
            config_hash = compute_config_hash(family, normalized)
            if config_hash in seen_hashes:
                return False
            signature = coarse_signature_key(family, normalized)
            if signature in seen_signatures and not allow_near_duplicate:
                return False
            if any(
                signature_distance(family, normalized, existing["config"]) <= request.max_near_duplicate_distance
                for existing in selection_pool
            ) and not allow_near_duplicate:
                return False
            seen_hashes.add(config_hash)
            seen_signatures.add(signature)
            candidate_metadata = _candidate_metadata(
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
                template_counts=template_counts or family_analysis.get("template_counts", {}),
                lineage_root_config_hash=lineage_root_config_hash,
                branch_budget=branch_budget,
                branch_budget_share=branch_budget_share,
                branch_budget_stance=branch_budget_stance,
                branch_budget_reason=branch_budget_reason,
                targeted_follow_up_required=targeted_follow_up_required,
                targeted_follow_up_reason=targeted_follow_up_reason,
                targeted_follow_up_type=targeted_follow_up_type,
                targeted_follow_up_priority=targeted_follow_up_priority,
                targeted_follow_up_batch_id=targeted_follow_up_batch_id,
                confirmation_state=confirmation_state,
                confirmation_required=confirmation_required,
                confirmation_reason=confirmation_reason,
                confirmation_batch_id=confirmation_batch_id,
                confirmation_trial_kind=confirmation_trial_kind,
                holdout_check_required=holdout_check_required,
                holdout_check_type=holdout_check_type,
                holdout_check_status=holdout_check_status,
                holdout_check_outcome=holdout_check_outcome,
                holdout_check_scope=holdout_check_scope,
                holdout_check_batch_id=holdout_check_batch_id,
                holdout_horizon_tags=holdout_horizon_tags,
                holdout_regime_tags=holdout_regime_tags,
                structural_novelty_threshold=structural_novelty_threshold,
                uncommon_template_bonus=uncommon_template_bonus,
                idea_id=idea_id,
                idea_source=idea_source,
                idea_kind=idea_kind,
                synthesized_template_family=synthesized_template_family,
                synthesis_rationale=synthesis_rationale,
            )
            candidate_metadata["confirmation_slot_index"] = len(selection_pool) + 1
            if _cycle_caps_exhausted(candidate_metadata):
                return False
            if _repeat_branch_exhausted(candidate_metadata):
                return False
            selection_pool.append(candidate_metadata)
            _record_repeat_branch(candidate_metadata)
            _record_cycle_caps(candidate_metadata)
            return True

        # Structural novelty floor: inject one guaranteed slot per protected momentum
        # template that is below _STRUCTURAL_EVIDENCE_THRESHOLD.  These slots are added
        # FIRST so that branch-budget / confirmation candidates cannot crowd them out.
        # One slot per behind-template per cycle; least-tested template gets priority.
        if family == "momentum" and _structural_behind:
            _struct_floor_payloads = [
                build_template_payload(t, family)
                for t in list_idea_templates(family)
                if t.template_id in _structural_behind and t.source_type != "cross_family_hybrid"
            ]
            _struct_floor_payloads.sort(
                key=lambda p: int(_structural_tc.get(p["metadata"]["template_id"], 0))
            )
            # Cap floor at at most 1/3 of budget so normal template slots remain available.
            _floor_cap = max(1, budget // 3)
            _seen_floor_tids: set[str] = set()
            for _sp in _struct_floor_payloads:
                if len(selection_pool) >= budget or len(_seen_floor_tids) >= _floor_cap:
                    break
                _sp_tid = _sp["metadata"].get("template_id") or ""
                if _sp_tid in _seen_floor_tids:
                    continue
                _seen_floor_tids.add(_sp_tid)
                _sp_execs = int(_structural_tc.get(_sp_tid, 0))
                _sm = _sp["metadata"]
                _structural_budget_share = round(
                    1.0 / max(1, len(_structural_behind)), 3
                )
                _try_add_candidate(
                    _sp["config"],
                    source_type=_sm.get("source_type") or "structural_novelty",
                    strategy_type=_sm.get("strategy_type") or _family_strategy_type(family),
                    template_id=_sp_tid,
                    hypothesis=_sm.get("hypothesis"),
                    reason_selected=(
                        f"structural floor: {_sp_tid} has {_sp_execs} execs "
                        f"(below evidence threshold {_STRUCTURAL_EVIDENCE_THRESHOLD}); "
                        f"structural_budget_share={_structural_budget_share}; "
                        f"behind_count={len(_structural_behind)}; "
                        f"handoff_state={_handoff_state}"
                    ),
                    exploration_mode=_sm.get("exploration_mode") or "structural_exploration",
                    proposal_role="explore",
                    region_label=_sp_tid,
                    source_idea_ids=[],
                    template_tags=_sm.get("tags"),
                    allow_near_duplicate=True,
                    novelty_floor_override=0.0,
                )

        # ── Family-discovery probe guaranteed floor ──────────────────────────
        # Reserved BEFORE confirmation candidates so that 1-2 fd-probe direct-synthesis
        # slots are always available even in confirmation-dominant modes.
        # Hard cap: _FD_PROBE_CONFIRMATION_BUDGET (=2); only fires when:
        #   - family-discovery probe ideas exist (idea_source=="family_discovery")
        #   - structural execution lane is enabled
        #   - pool has remaining capacity
        # Uses allow_near_duplicate=False to preserve deterministic safety.
        _FD_PROBE_CONFIRMATION_BUDGET = 2
        _fd_probe_budget_used = 0
        _fd_probe_lane_entries: list[dict[str, Any]] = []

        if fd_probe_ideas and structural_execution_lane_enabled and len(selection_pool) < budget:
            _fd_synth_candidates = _synthesized_structural_candidates(
                family=family,
                ideas=fd_probe_ideas,
                idea_yield=_fam_yield,
                recent_idea_signatures=set(),
                limit=_FD_PROBE_CONFIRMATION_BUDGET * 3,  # request more; many may be blocked
            )
            for _fd_payload in _fd_synth_candidates:
                if _fd_probe_budget_used >= _FD_PROBE_CONFIRMATION_BUDGET:
                    break
                if len(selection_pool) >= budget:
                    break
                _fd_meta = _fd_payload["metadata"]
                _fd_idea = _fd_payload.get("idea") or {}
                _fd_idea_id = str(_fd_meta.get("idea_id") or "")
                _fd_candidate_id = str((_fd_idea.get("metadata") or {}).get("family_candidate_id") or "")
                _fd_family_name = str(
                    (_fd_idea.get("metadata") or {}).get("discovery_family_name")
                    or (_fd_idea.get("suggested_config") or {}).get("family_discovery_candidate")
                    or _fd_idea_id
                )
                _fd_synth_class = str(_fd_meta.get("synthesized_template_family") or "structural_extension")
                _fd_tmpl_class = str(_fd_idea.get("template_similarity_class") or _fd_synth_class)
                # Try the base config plus each variant override until one passes.
                # This avoids being permanently blocked by an already-executed baseline config.
                _fd_base_config = _fd_payload["config"]
                _fd_variant_overrides = _FD_PROBE_VARIANT_OVERRIDES.get(_fd_synth_class, [{}])
                _fd_added = False
                for _fd_variant_idx, _fd_override in enumerate(_fd_variant_overrides):
                    _fd_cfg = {**_fd_base_config, **_fd_override} if _fd_override else _fd_base_config
                    _fd_added = _try_add_candidate(
                        _fd_cfg,
                        source_type="family_discovery_direct_synthesis",
                        strategy_type=_fd_meta.get("strategy_type") or _family_strategy_type(family),
                        template_id=_fd_meta.get("template_id"),
                        hypothesis=_fd_meta.get("hypothesis"),
                        reason_selected=(
                            f"fd_probe_floor: family={_fd_family_name}; "
                            f"candidate_id={_fd_candidate_id}; "
                            f"synth_class={_fd_synth_class}; "
                            f"template_class={_fd_tmpl_class}; "
                            f"idea_id={_fd_idea_id}; "
                            f"variant={_fd_variant_idx}; "
                            f"confirmation_mode_reserved={family_confirmation_mode}"
                        ),
                        exploration_mode="family_discovery_probe",
                        proposal_role="explore",
                        region_label=f"fd_probe_{_fd_family_name[:24]}",
                        source_idea_ids=[_fd_idea_id] if _fd_idea_id else [],
                        idea_id=_fd_idea_id,
                        idea_source=_fd_idea.get("idea_source") or "family_discovery",
                        idea_kind=_fd_idea.get("idea_kind") or "new_family_discovery_probe",
                        template_tags=list(_fd_meta.get("template_tags") or []),
                        template_counts=family_analysis.get("template_counts", {}),
                        structural_novelty_threshold=request.structural_novelty_threshold,
                        uncommon_template_bonus=request.uncommon_template_bonus,
                        synthesized_template_family=_fd_synth_class,
                        synthesis_rationale=(
                            f"family_discovery_direct_synthesis: "
                            f"family={_fd_family_name}; "
                            f"candidate_id={_fd_candidate_id}; "
                            f"template_class={_fd_tmpl_class}; "
                            f"synth_family={_fd_synth_class}; "
                            f"idea_source=family_discovery; "
                            f"variant={_fd_variant_idx}; "
                            f"confirmation_mode_reserved={family_confirmation_mode}"
                        ),
                        allow_near_duplicate=False,
                        novelty_floor_override=0.0,
                    )
                    if _fd_added:
                        break
                if _fd_added:
                    _fd_probe_budget_used += 1
                    _fd_probe_lane_entries.append({
                        "idea_id": _fd_idea_id,
                        "family_name": _fd_family_name,
                        "candidate_id": _fd_candidate_id,
                        "synth_class": _fd_synth_class,
                        "template_class": _fd_tmpl_class,
                        "is_confirmation_reserved": family_confirmation_mode,
                        "variant_index": _fd_variant_idx,
                    })

        if family_confirmation_mode and not history.empty:
            _append_confirmation_reproduction(top.iloc[0])
            _append_confirmation_neighborhood(top.iloc[0])
            confirmation_diversity_floor = min(
                confirmation_trial_budget,
                max(4, int(round(max(1, confirmation_trial_budget) * 0.40))),
            )
            if len(selection_pool) < confirmation_diversity_floor:
                # Keep the targeted confirmation candidate, but do not let an
                # exhausted local neighborhood starve the rest of the cycle.
                confirmation_focus_only = False

        structural_lane_state = {
            "active": False,
            "planner_requested": bool(structural_execution_lane_enabled),
            "candidate_count": len(structural_family_ideas),
            "effective_budget": 0,
            "executed_count": 0,
            "skipped": False,
            "skip_reason": None,
            "reason": None,
            "recent_active": bool(_fam_structural_promotion_state in {"floor_protected", "yield_supported", "graduated_structural_family"} or structural_yield_records),
            "recent_family_counts": {},
            "recent_llm_seeded_count": sum(1 for record in structural_family_ideas if list(record.get("source_idea_ids") or [])),
        }
        if structural_execution_lane_enabled and structural_family_ideas:
            structural_lane_budget = min(
                budget,
                structural_execution_lane_budget,
            )
            if structural_lane_budget <= 0:
                structural_lane_state["skipped"] = True
                structural_lane_state["skip_reason"] = "lane_budget_zero"
                structural_lane_state["reason"] = "deterministic lane budget reduced to zero"
            else:
                structural_lane_state["active"] = True
                structural_lane_state["effective_budget"] = structural_lane_budget
                structural_candidates = _synthesized_structural_candidates(
                    family=family,
                    ideas=structural_family_ideas,
                    idea_yield=_fam_yield,
                    recent_idea_signatures=set(),
                    limit=structural_lane_budget,
                )
                structural_family_counts: dict[str, int] = {}
                structural_budget_counts = {"structural": 0, "out_of_box": 0, "uncommon": 0}
                structural_family_caps = {
                    synth_family: max(1, min(structural_execution_lane_repeat_cap, structural_family_targets.get(synth_family, 1)))
                    for synth_family in structural_family_targets
                }
                for payload in structural_candidates:
                    if len(selection_pool) >= budget or structural_budget_counts["structural"] >= structural_lane_budget:
                        break
                    metadata = payload["metadata"]
                    synth_family = str(metadata.get("synthesized_template_family") or "structural_extension")
                    family_cap = structural_family_caps.get(synth_family, max(1, structural_execution_lane_repeat_cap))
                    if structural_family_counts.get(synth_family, 0) >= family_cap:
                        continue
                    if bool(metadata.get("is_out_of_box")) and structural_budget_counts["out_of_box"] >= out_of_box_budget:
                        continue
                    if bool(metadata.get("is_uncommon_idea")) and structural_budget_counts["uncommon"] >= uncommon_idea_budget:
                        continue
                    added = _try_add_candidate(
                        payload["config"],
                        source_type="structural_synthesis",
                        strategy_type=_family_strategy_type(family),
                        template_id=metadata.get("template_id"),
                        hypothesis=metadata.get("hypothesis"),
                        reason_selected=metadata.get("reason_selected"),
                        exploration_mode="structural_synthesis",
                        proposal_role="explore",
                        region_label=synth_family,
                        source_idea_ids=list(metadata.get("source_idea_ids") or []),
                        idea_id=metadata.get("idea_id"),
                        idea_source=metadata.get("idea_source"),
                        idea_kind=metadata.get("idea_kind"),
                        template_tags=list(metadata.get("template_tags") or []),
                        template_counts=family_analysis.get("template_counts", {}),
                        structural_novelty_threshold=request.structural_novelty_threshold,
                        uncommon_template_bonus=request.uncommon_template_bonus,
                        synthesized_template_family=synth_family,
                        synthesis_rationale=metadata.get("synthesis_rationale"),
                        allow_near_duplicate=False,
                    )
                    if added:
                        structural_budget_counts["structural"] += 1
                        structural_family_counts[synth_family] = structural_family_counts.get(synth_family, 0) + 1
                        if bool(metadata.get("is_out_of_box")):
                            structural_budget_counts["out_of_box"] += 1
                        if bool(metadata.get("is_uncommon_idea")):
                            structural_budget_counts["uncommon"] += 1
                structural_lane_state["executed_count"] = structural_budget_counts["structural"]
                structural_lane_state["recent_family_counts"] = dict(structural_family_counts)
                if structural_lane_state["executed_count"] <= 0:
                    structural_lane_state["skipped"] = True
                    structural_lane_state["skip_reason"] = "no_safe_structural_candidate"
                    structural_lane_state["reason"] = "structural lane was available but no candidate passed deterministic safety checks"
                elif structural_lane_state["executed_count"] < structural_execution_lane_min_candidates:
                    structural_lane_state["reason"] = (
                        "structural lane executed below the requested floor because deterministic caps "
                        "limited the safe candidate count"
                    )
                else:
                    structural_lane_state["reason"] = (
                        "structural lane executed within its protected budget slice and remained under deterministic caps"
                    )
        else:
            structural_lane_state["skipped"] = True
            structural_lane_state["skip_reason"] = "disabled_or_no_structural_candidates"
            structural_lane_state["reason"] = (
                "structural lane disabled by plan or no structural-synthesis candidates were available"
            )

        # Build fd probe lane state for reporting
        fd_probe_lane_state: dict[str, Any] = {
            "enabled": bool(fd_probe_ideas and structural_execution_lane_enabled),
            "available_ideas": len(fd_probe_ideas),
            "reservation_budget": _FD_PROBE_CONFIRMATION_BUDGET,
            "executed_count": _fd_probe_budget_used,
            "confirmation_mode": family_confirmation_mode,
            "entries": _fd_probe_lane_entries,
            "skipped": _fd_probe_budget_used == 0,
            "skip_reason": (
                "no_fd_probe_ideas" if not fd_probe_ideas
                else "structural_lane_disabled" if not structural_execution_lane_enabled
                else "pool_full_before_fd_floor" if len(selection_pool) <= 0
                else "no_safe_candidate_passed" if _fd_probe_budget_used == 0
                else None
            ),
        }

        if family_branch_budget_entries and not confirmation_mode:
            for branch_entry in family_branch_budget_entries:
                if len(selection_pool) >= budget:
                    break
                branch_seed_row = _branch_seed_row(history, branch_entry)
                if branch_seed_row is None:
                    continue
                branch_limit = max(1, min(int(branch_entry.get("branch_budget") or 0), budget - len(selection_pool)))
                branch_neighbors = build_local_neighbors(
                    family,
                    dict(branch_seed_row.get("config") or {}),
                    limit=branch_limit,
                    seed=request.seed + proposal_seed_offset + idx + len(selection_pool) + branch_limit,
                    dead_zones=dead_zones,
                    explored_hashes=seen_hashes,
                )
                for neighbor in branch_neighbors:
                    if len(selection_pool) >= budget:
                        break
                    branch_stance = str(branch_entry.get("branch_budget_stance") or "")
                    _try_add_candidate(
                        neighbor,
                        source_type="template_expansion",
                        strategy_type=_family_strategy_type(family),
                        template_id=branch_entry.get("branch_root_config_hash") or "lineage_branch",
                        hypothesis="Refine a promising lineage branch.",
                        reason_selected=branch_entry.get("branch_budget_reason")
                        or "lineage branch budget prioritizes a promising descendant branch",
                        exploration_mode="branch_refinement",
                        proposal_role="exploit",
                        region_label=branch_entry.get("branch_root_config_hash") or "lineage_branch",
                        parent_config_hash=branch_seed_row.get("config_hash"),
                        source_idea_ids=[],
                        template_tags=["lineage_branch"],
                        lineage_root_config_hash=branch_entry.get("branch_root_config_hash") or branch_seed_row.get("config_hash"),
                        branch_budget=int(branch_entry.get("branch_budget") or 0),
                        branch_budget_share=float(branch_entry.get("branch_budget_share") or 0.0),
                        branch_budget_stance=branch_stance,
                        branch_budget_reason=branch_entry.get("branch_budget_reason"),
                        novelty_floor_override=0.0,
                        allow_near_duplicate=branch_stance in {"primary", "controlled"},
                    )

        routine_template_count = 0
        if not confirmation_focus_only and routine_template_count < max_routine_template_budget:
            for idea in routine_family_ideas:
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
                    seed=request.seed + proposal_seed_offset + idx + len(selection_pool),
                    dead_zones=dead_zones,
                    explored_hashes=seen_hashes,
                )
                for neighbor in neighbors:
                    if len(selection_pool) >= budget:
                        break
                    if routine_template_count >= max_routine_template_budget:
                        break
                    added = _try_add_candidate(
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
                    if added:
                        routine_template_count += 1
                if routine_template_count >= max_routine_template_budget:
                    break

        for top_idx, row in top.iterrows():
            if len(selection_pool) >= exploit_n:
                break
            config = row.get("config") or {}
            neighbors = build_local_neighbors(
                family,
                config,
                limit=max(1, exploit_n - len(selection_pool)),
                seed=request.seed + proposal_seed_offset + idx + top_idx,
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
                    lineage_root_config_hash=row.get("lineage_root_config_hash") or row.get("config_hash"),
                    branch_budget=(
                        next(
                            (
                                int(entry.get("branch_budget") or 0)
                                for entry in family_branch_budget_entries
                                if str(entry.get("branch_root_config_hash") or "")
                                == str(row.get("lineage_root_config_hash") or row.get("config_hash") or "")
                            ),
                            None,
                        )
                        if family_branch_budget_entries
                        else None
                    ),
                    branch_budget_share=(
                        next(
                            (
                                float(entry.get("branch_budget_share") or 0.0)
                                for entry in family_branch_budget_entries
                                if str(entry.get("branch_root_config_hash") or "")
                                == str(row.get("lineage_root_config_hash") or row.get("config_hash") or "")
                            ),
                            None,
                        )
                        if family_branch_budget_entries
                        else None
                    ),
                    branch_budget_stance=(
                        next(
                            (
                                str(entry.get("branch_budget_stance") or "")
                                for entry in family_branch_budget_entries
                                if str(entry.get("branch_root_config_hash") or "")
                                == str(row.get("lineage_root_config_hash") or row.get("config_hash") or "")
                            ),
                            None,
                        )
                        if family_branch_budget_entries
                        else None
                    ),
                    branch_budget_reason=(
                        next(
                            (
                                str(entry.get("branch_budget_reason") or "")
                                for entry in family_branch_budget_entries
                                if str(entry.get("branch_root_config_hash") or "")
                                == str(row.get("lineage_root_config_hash") or row.get("config_hash") or "")
                            ),
                            None,
                        )
                        if family_branch_budget_entries
                        else None
                    ),
                    allow_near_duplicate=False,
                )

        if not confirmation_focus_only and routine_template_count < max_routine_template_budget:
            template_payloads = _template_payloads_for_family(
                family,
                limit=max(template_n + cross_family_n + 4, 8),
                seed=request.seed + proposal_seed_offset + 1000 + idx,
                allow_cross_family=True,
            )
            template_payloads.sort(key=lambda item: item["metadata"].get("source_type") != "cross_family_hybrid")
            template_count = 0
            cross_count = 0
            for payload in template_payloads:
                if len(selection_pool) >= budget:
                    break
                if routine_template_count >= max_routine_template_budget:
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
                added = _try_add_candidate(
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
                if added:
                    routine_template_count += 1

            _top_configs_for_smart_search = [
                dict(row.get("config") or {})
                for _, row in top.iterrows()
                if row.get("config")
            ] if not top.empty else []
            _smart_search_meta: dict[str, Any] = {}
            exploration_candidates = sample_exploration_configs(
                family,
                limit=max(local_explore_n * 4, local_explore_n, 4),
                seed=request.seed + proposal_seed_offset + 2000 + idx,
                explored_hashes=seen_hashes,
                dead_zones=dead_zones,
                cycle_mode=lineage_cycle_mode,
                history_count=len(history),
                stagnation_batches=stagnation_batches,
                top_configs=_top_configs_for_smart_search,
                out_metadata=_smart_search_meta,
            )
            reasoning["families"].setdefault(family, {}).setdefault("search_meta", {}).update(_smart_search_meta)
            for candidate in exploration_candidates:
                if len(selection_pool) >= budget:
                    break
                if routine_template_count >= max_routine_template_budget:
                    break
                added = _try_add_candidate(
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
                if added:
                    routine_template_count += 1

            if request.allow_external_seeds:
                for seed_payload in load_external_idea_seeds(enabled=True):
                    if len(selection_pool) >= budget:
                        break
                    if routine_template_count >= max_routine_template_budget:
                        break
                    added = _try_add_candidate(
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
                    if added:
                        routine_template_count += 1

            if len(selection_pool) < budget:
                fallback = sample_exploration_configs(
                    family,
                    limit=budget - len(selection_pool),
                    seed=request.seed + proposal_seed_offset + 3000 + idx,
                    explored_hashes=seen_hashes,
                    dead_zones={},
                    cycle_mode=lineage_cycle_mode,
                    history_count=len(history),
                    stagnation_batches=stagnation_batches,
                    top_configs=_top_configs_for_smart_search,
                )
                for candidate in fallback:
                    if len(selection_pool) >= budget:
                        break
                    if routine_template_count >= max_routine_template_budget:
                        break
                    added = _try_add_candidate(
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
                    if added:
                        routine_template_count += 1

            if len(selection_pool) < budget:
                saturation_escape = sample_exploration_configs(
                    family,
                    limit=budget - len(selection_pool),
                    seed=request.seed + proposal_seed_offset + 4000 + idx,
                    explored_hashes=seen_hashes,
                    dead_zones={},
                    sample_multiplier=64,
                    cycle_mode="saturation_escape",
                    history_count=len(history),
                    stagnation_batches=stagnation_batches,
                    top_configs=_top_configs_for_smart_search,
                )
                for candidate in saturation_escape:
                    if len(selection_pool) >= budget:
                        break
                    if routine_template_count >= max_routine_template_budget:
                        break
                    added = _try_add_candidate(
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
                    if added:
                        routine_template_count += 1
        else:
            if holdout_check_required and not history.empty:
                _append_confirmation_neighborhood(top.iloc[0], holdout=True)
            exploration_candidates = sample_exploration_configs(
                family,
                limit=max(confirmation_trial_budget - len(selection_pool), 1),
                seed=request.seed + proposal_seed_offset + 2000 + idx,
                explored_hashes=seen_hashes,
                dead_zones=dead_zones,
                cycle_mode="confirmation",
                history_count=len(history),
                stagnation_batches=stagnation_batches,
                top_configs=[
                    dict(row.get("config") or {})
                    for _, row in top.iterrows()
                    if row.get("config")
                ] if not top.empty else [],
            )
            for candidate in exploration_candidates:
                if len(selection_pool) >= confirmation_trial_budget:
                    break
                _try_add_candidate(
                    candidate,
                    source_type="confirmation_check",
                    strategy_type=_family_strategy_type(family),
                    template_id="confirmation_robustness_check",
                    hypothesis="Robustness check around the promoted winner.",
                    reason_selected="confirmation batch robustness check",
                    exploration_mode="confirmation",
                    proposal_role="confirm",
                    region_label="confirmation_robustness_check",
                    source_idea_ids=[],
                    allow_near_duplicate=False,
                )

        ranked_payloads = sorted(
            selection_pool,
            key=lambda item: (
                -int(str(item.get("source_type") or "") == "structural_synthesis"),
                -int(bool(item.get("synthesized_template_family"))),
                -int(bool(item.get("branch_budget_stance"))),
                -float(item.get("branch_budget") or 0.0),
                -int(bool(item.get("is_uncommon_idea"))),
                -int(bool(item.get("is_structurally_novel"))),
                -int(bool(item.get("is_new_idea"))),
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
        selected_structural = [item for item in candidate_metadata[family] if item.get("source_type") == "structural_synthesis"]
        selected_routine = [
            item
            for item in candidate_metadata[family]
            if item.get("source_type")
            in {"idea_seed", "template_expansion", "cross_family_hybrid", "broad_exploration", "external_seed", "saturation_escape"}
            and not item.get("branch_budget_stance")
            and not item.get("branch_budget")
        ]
        reasoning["families"][family] = {
            "baseline_name": family_analysis.get("baseline_name"),
            "history_count": family_analysis["history_count"],
            "history_limit_per_family": history_limit_per_family,
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
            "branch_budgets": branch_budgets.get(family) or [],
            "branch_budget_rationale": (branch_budget_rationale.get("families") or {}).get(family, {}),
            "new_idea_budget": request.new_idea_budget,
            "refinement_budget": request.refinement_budget,
            "confirmation_budget": request.confirmation_budget,
            "new_idea_quota": request.new_idea_quota,
            "uncommon_idea_quota": request.uncommon_idea_quota,
            "repeat_branch_cap": request.repeat_branch_cap,
            "max_same_template_per_cycle": request.max_same_template_per_cycle,
            "max_same_lineage_per_cycle": request.max_same_lineage_per_cycle,
            "structural_novelty_threshold": request.structural_novelty_threshold,
            "uncommon_template_bonus": request.uncommon_template_bonus,
            "uncommon_idea_floor": cycle_uncommon_idea_floor,
            "exploit_count": min(len(candidate_configs[family]), exploit_n),
            "explore_count": max(0, len(candidate_configs[family]) - min(len(candidate_configs[family]), exploit_n)),
            "exploration_fraction": exploration_fraction,
            "exploitation_fraction": exploitation_fraction,
            "deprioritized_regions": family_analysis["dead_zones"],
            "baseline_win_rate": family_analysis["baseline_win_rate"],
            "novelty_floor": novelty_floor,
            "selection_count": len(candidate_configs[family]),
            "confirmation_state": request.confirmation_state,
            "confirmation_required": bool(request.confirmation_required),
            "confirmation_reason": request.confirmation_reason,
            "confirmation_batch_id": request.confirmation_batch_id,
            "confirmation_trial_budget": confirmation_trial_budget if family_confirmation_mode else None,
            "confirmation_focus_family": confirmation_focus_family,
            "targeted_follow_up_required": targeted_follow_up_mode,
            "targeted_follow_up_reason": targeted_follow_up_reason,
            "targeted_follow_up_type": targeted_follow_up_type,
            "targeted_follow_up_priority": targeted_follow_up_priority,
            "targeted_follow_up_batch_id": targeted_follow_up_batch_id,
            "holdout_check_required": holdout_check_required,
            "holdout_check_type": holdout_check_type,
            "holdout_check_status": holdout_check_status,
            "holdout_check_outcome": holdout_check_outcome,
            "holdout_check_scope": holdout_check_scope,
            "holdout_check_batch_id": holdout_check_batch_id,
            "holdout_horizon_tags": holdout_horizon_tags,
            "holdout_regime_tags": holdout_regime_tags,
            "idea_yield_state": _fam_idea_state,
            "idea_yield_quality": round(_fam_idea_quality, 4),
            "idea_yield_retired_share": round(_fam_retired_share, 4),
            "idea_yield_promising_share": round(_fam_promising_share, 4),
            "structural_family_state": _fam_structural_state,
            "structural_family_state_previous": _fam_structural_state_previous,
            "structural_family_promotion_state": _fam_structural_promotion_state,
            "structural_family_transition_reason": _fam_structural_transition_reason,
            "graduated_template_family": _fam_yield.get("graduated_template_family"),
            "structural_family_feedback": structural_feedback,
            "idea_yield_floor_action": _yield_floor_action,
            "idea_yield_top_kinds": [r.get("idea_kind") for r in (_fam_yield.get("top_idea_kinds") or [])[:3] if r.get("idea_kind")],
            "structural_idea_budget": structural_idea_budget,
            "structural_execution_lane_enabled": structural_execution_lane_enabled,
            "structural_execution_lane_budget": structural_execution_lane_budget,
            "structural_execution_lane_min_candidates": structural_execution_lane_min_candidates,
            "structural_execution_lane_repeat_cap": structural_execution_lane_repeat_cap,
            "out_of_box_budget": out_of_box_budget,
            "uncommon_idea_budget": uncommon_idea_budget,
            "max_routine_template_budget": max_routine_template_budget,
            "structural_budget_state": structural_budget_state,
            "structural_candidate_count": len(structural_family_ideas),
            "structural_successful_families": sorted(structural_successful_families),
            "structural_family_targets": dict(structural_family_targets),
            "structural_execution_lane": structural_lane_state,
            "fd_probe_lane": fd_probe_lane_state,
            "fd_probe_executed_count": _fd_probe_budget_used,
            "fd_probe_confirmation_reserved": family_confirmation_mode and _fd_probe_budget_used > 0,
            "structural_execution_lane_active_recent_cycles": bool(structural_lane_state.get("recent_active")),
            "structural_execution_lane_planner_requested": bool(structural_lane_state.get("planner_requested")),
            "structural_execution_lane_received_candidates": int(structural_lane_state.get("candidate_count") or 0),
            "structural_execution_lane_executed_count": int(structural_lane_state.get("executed_count") or 0),
            "structural_execution_lane_skipped": bool(structural_lane_state.get("skipped")),
            "structural_execution_lane_skip_reason": structural_lane_state.get("skip_reason"),
            "routine_template_budget_used": routine_template_count,
            "selected_structural_synthesis_count": len(selected_structural),
            "selected_routine_candidate_count": len(selected_routine),
            "selected_structural_synthesis_families": sorted(
                {
                    str(item.get("synthesized_template_family") or "")
                    for item in selected_structural
                    if item.get("synthesized_template_family")
                }
            ),
            "structural_handoff_state": _handoff_state if family == "momentum" else None,
            "structural_handoff_rationale": _handoff_rationale if family == "momentum" else None,
            "structural_yield_scores": {k: round(v, 4) for k, v in _structural_yield_scores.items()} if family == "momentum" else {},
            "structural_per_template_cap": dict(_structural_per_template_cap) if family == "momentum" else {},
            "adaptive_template_fraction": _eff_template_frac,
            "adaptive_cross_family_fraction": _eff_cf_frac,
            "branch_dominance": round(_branch_dominance, 4),
        }
        if isinstance(family_analysis.get("history_frame"), pd.DataFrame):
            family_analysis["history_frame"] = pd.DataFrame()
        try:
            del history
        except NameError:
            pass
        try:
            del history_records
        except NameError:
            pass
        try:
            del top
        except NameError:
            pass
        try:
            del robust
        except NameError:
            pass
        try:
            del unstable
        except NameError:
            pass
        try:
            del exact_baseline
        except NameError:
            pass
        gc.collect()

    proposal_memory = load_research_memory(base_dir)
    for family in request.strategy_families:
        family_analysis = analysis["families"][family]
        exact_hashes = set(family_analysis["explored_hashes"])
        signatures = set(family_analysis["explored_signatures"].values())
        dead_zones = {key: set(values) for key, values in (family_analysis.get("dead_zones") or {}).items()}
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
        if isinstance(family_analysis.get("history_frame"), pd.DataFrame):
            family_analysis["history_frame"] = pd.DataFrame()
        gc.collect()
    if request.persist_memory:
        save_research_memory(proposal_memory, base_dir)
    if request.persist_scorecards:
        scorecards = build_family_scorecards(families=request.strategy_families, base_dir=base_dir)
        scorecard_path = save_family_scorecards(scorecards, base_dir=base_dir, timestamp_utc=request.timestamp_utc)
        reasoning["family_scorecard_path"] = str(scorecard_path)

    # Persist a fresh idea-yield summary so future cycles can load it via load_latest_idea_yield_summary().
    try:
        _fresh_yield_summary = build_idea_yield_summary(
            families=request.strategy_families, base_dir=base_dir
        )
        _yield_report_path = save_idea_yield_summary(_fresh_yield_summary, base_dir=base_dir)
        reasoning["idea_yield_report_path"] = str(_yield_report_path)
    except Exception:
        pass  # Non-fatal: yield tracking is advisory only

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
    if not request.persist_proposal:
        return ProposalResult(
            request=proposal.request,
            status=proposal.status,
            candidate_configs=proposal.candidate_configs,
            candidate_metadata=proposal.candidate_metadata,
            reasoning_summary=proposal.reasoning_summary,
            proposal_path=None,
            summary_path=None,
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
