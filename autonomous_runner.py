#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace as dc_replace
from pathlib import Path
from typing import Any

from agents.schemas import load_latest_pending_proposal_record, update_proposal_record_status

from experiment_batch import (
    build_batch_leaderboard,
    build_batch_request,
    proposal_to_batch_request,
    run_batch_experiments,
)
from experiment_memory_guard import current_process_memory_kb
from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_runtime_decision import build_runtime_decision
from experiment_spaces import list_searchable_families
from experiment_types import ProposalRequest, ProposalResult, RuntimeDecision, RuntimeDecisionInput

# RSS thresholds for memory pressure caps
_RSS_ELEVATED_KB = 12_000_000  # 12 GB
_LOG_MAX_LIST_ITEMS = 3
_LOG_MAX_DICT_ITEMS = 5
_LOG_MAX_STRING_LEN = 120
_LEADERBOARD_PREVIEW_ROWS = 3
_LEADERBOARD_PREVIEW_COLUMNS = (
    "experiment_id",
    "strategy_family",
    "objective_score",
    "viable",
    "beats_baseline_objective",
    "comparison_status",
)


def _truncate_text(value: str, *, max_len: int = _LOG_MAX_STRING_LEN) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


def _format_compact_scalar(value: Any) -> str:
    if isinstance(value, str):
        return repr(_truncate_text(value))
    return repr(value)


def _format_compact_list(values: Any, *, max_items: int = _LOG_MAX_LIST_ITEMS) -> str:
    if not isinstance(values, (list, tuple)):
        return _format_compact_scalar(values)
    items = [
        _format_compact_scalar(item)
        for item in list(values)[:max_items]
    ]
    remaining = len(values) - len(items)
    if remaining > 0:
        items.append(f"... +{remaining} more")
    return "[" + ", ".join(items) + "]"


def _format_compact_dict(values: Any, *, max_items: int = _LOG_MAX_DICT_ITEMS) -> str:
    if not isinstance(values, dict):
        return _format_compact_scalar(values)
    parts = []
    for index, (key, value) in enumerate(values.items()):
        if index >= max_items:
            break
        parts.append(f"{key}={_format_compact_value(value)}")
    remaining = len(values) - len(parts)
    if remaining > 0:
        parts.append(f"... +{remaining} more keys")
    return "{" + ", ".join(parts) + "}"


def _format_compact_value(value: Any) -> str:
    if isinstance(value, dict):
        return _format_compact_dict(value)
    if isinstance(value, (list, tuple)):
        return _format_compact_list(value)
    if isinstance(value, str):
        return repr(_truncate_text(value))
    return repr(value)


def _format_selected_families(families: Any) -> str:
    if not isinstance(families, (list, tuple)):
        return _format_compact_value(families)
    return f"{len(families)} {_format_compact_list(families)}"


def _format_reasoning_summary(summary: Any) -> str:
    if not isinstance(summary, dict):
        return _format_compact_value(summary)

    parts: list[str] = []
    quality = summary.get("proposal_quality")
    if isinstance(quality, dict):
        fill = f"{quality.get('candidate_count')}/{quality.get('requested')}"
        parts.append(
            "quality["
            f"status={quality.get('status')} "
            f"score={quality.get('quality_score')} "
            f"fill={fill} "
            f"min_viable={quality.get('min_viable_candidates')}"
            "]"
        )
        if quality.get("shortfall_reasons"):
            parts.append(f"quality_shortfall={_format_compact_list(quality.get('shortfall_reasons'))}")

    runtime_decision = summary.get("runtime_decision")
    if isinstance(runtime_decision, dict):
        parts.append(
            "runtime["
            f"decision_id={runtime_decision.get('decision_id')} "
            f"mode={runtime_decision.get('cycle_mode')} "
            f"fallback={runtime_decision.get('fallback_used')} "
            f"families={_format_selected_families(runtime_decision.get('selected_families'))} "
            f"confirmation={runtime_decision.get('confirmation_required')} "
            f"holdout={runtime_decision.get('holdout_check_required')}"
            "]"
        )

    structural_lane = summary.get("structural_execution_lane")
    if isinstance(structural_lane, dict):
        parts.append(
            "structural_lane["
            f"active={structural_lane.get('active')} "
            f"planner_requested={structural_lane.get('planner_requested')} "
            f"budget={structural_lane.get('effective_budget')} "
            f"received={structural_lane.get('candidate_count')} "
            f"executed={structural_lane.get('executed_count')} "
            f"skipped={structural_lane.get('skipped')} "
            f"reason={_format_compact_scalar(structural_lane.get('reason'))}"
            "]"
        )
    else:
        families = summary.get("families")
        family_lanes: list[tuple[str, dict[str, Any]]] = []
        if isinstance(families, dict):
            for family_name, family_summary in families.items():
                if not isinstance(family_summary, dict):
                    continue
                lane = family_summary.get("structural_execution_lane")
                if isinstance(lane, dict):
                    family_lanes.append((str(family_name), lane))
        if family_lanes:
            active_count = sum(1 for _, lane in family_lanes if lane.get("active"))
            skipped_count = sum(1 for _, lane in family_lanes if lane.get("skipped"))
            requested_count = sum(1 for _, lane in family_lanes if lane.get("planner_requested"))
            received_count = sum(int(lane.get("candidate_count") or 0) for _, lane in family_lanes)
            executed_count = sum(int(lane.get("executed_count") or 0) for _, lane in family_lanes)
            deterministic_override = any(
                bool(lane.get("skipped")) or int(lane.get("executed_count") or 0) < int(lane.get("candidate_count") or 0)
                for _, lane in family_lanes
            )
            skip_reason = next((lane.get("skip_reason") for _, lane in family_lanes if lane.get("skip_reason")), None)
            reason = next((lane.get("reason") for _, lane in family_lanes if lane.get("reason")), None)
            parts.append(
                "structural_lane["
                f"families={len(family_lanes)} "
                f"requested={requested_count} "
                f"active={active_count} "
                f"received={received_count} "
                f"executed={executed_count} "
                f"skipped={skipped_count} "
                f"deterministic_override={deterministic_override} "
                f"skip_reason={_format_compact_scalar(skip_reason)} "
                f"reason={_format_compact_scalar(reason)}"
                "]"
            )

    analysis_provenance = summary.get("analysis_provenance")
    if isinstance(analysis_provenance, dict):
        batch_ids = analysis_provenance.get("batch_ids") or []
        parts.append(
            "provenance["
            f"batch_ids={len(batch_ids)} {_format_compact_list(batch_ids)}"
            "]"
        )
        idea_ids = analysis_provenance.get("source_idea_ids") or analysis_provenance.get("idea_ids") or []
        if idea_ids:
            parts.append(f"ideas={len(idea_ids)} {_format_compact_list(idea_ids)}")

    planning_rationale = summary.get("planning_rationale")
    if planning_rationale:
        parts.append(f"planning_rationale={_format_compact_value(planning_rationale)}")

    layer2_plan_id = summary.get("layer2_plan_id")
    if layer2_plan_id:
        parts.append(f"layer2_plan_id={layer2_plan_id}")

    if not parts:
        return _format_compact_dict(summary)
    return " | ".join(parts)


def _format_leaderboard_preview(leaderboard: Any, *, max_rows: int = 5) -> str:
    if getattr(leaderboard, "empty", True):
        return ""
    columns = [
        column for column in _LEADERBOARD_PREVIEW_COLUMNS if column in leaderboard.columns
    ]
    preview = leaderboard.head(max_rows)
    lines = [f"leaderboard_preview top={len(preview)}/{len(leaderboard)} cols={len(columns)}"]
    for row in preview.to_dict(orient="records"):
        parts = []
        for column in columns:
            value = row.get(column)
            if column == "objective_score" and value is not None:
                try:
                    parts.append(f"{column}={float(value):.3f}")
                    continue
                except (TypeError, ValueError):
                    pass
            parts.append(f"{column}={_format_compact_value(value)}")
        lines.append("  " + ", ".join(parts))
    remaining = len(leaderboard) - len(preview)
    if remaining > 0:
        lines.append(f"  ... +{remaining} more rows")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--family", default="all", help="momentum, superstock, or all")
    parser.add_argument("--method", default="random", choices=["random", "grid"], help="Sampling method")
    parser.add_argument("--n", type=int, default=24, help="Maximum experiments to run")
    parser.add_argument("--max-per-family", type=int, default=None, help="Maximum sampled configs per family")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-workers", type=int, default=6, help="Parallel worker count (max 8)")
    parser.add_argument("--base-dir", default="experiments", help="Experiment store root")
    parser.add_argument("--no-persist", action="store_true", help="Do not persist single-experiment results")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip prior successful configs")
    parser.add_argument("--proposal-next", action="store_true", help="Generate the next-round proposal from history")
    parser.add_argument("--run-proposal", action="store_true", help="Run the generated proposal through the batch runner")
    parser.add_argument("--exploration-fraction", type=float, default=0.65, help="Proposal exploration fraction")
    parser.add_argument("--exploitation-fraction", type=float, default=0.35, help="Proposal exploitation fraction")
    parser.add_argument("--novelty-floor", type=float, default=0.15, help="Minimum novelty score for proposed ideas")
    parser.add_argument("--template-fraction", type=float, default=0.50, help="Share of exploration budget for templates")
    parser.add_argument("--cross-family-fraction", type=float, default=0.20, help="Share of exploration budget for cross-family ideas")
    parser.add_argument(
        "--max-near-duplicate-distance",
        type=int,
        default=1,
        help="Maximum signature distance allowed for near-duplicate suppression",
    )
    parser.add_argument(
        "--stagnation-escape-batches",
        type=int,
        default=3,
        help="Widen exploration if no improvement occurs for this many batches",
    )
    parser.add_argument("--no-helper-ideas", action="store_true", help="Ignore queued helper ideas during proposal generation")
    parser.add_argument("--no-analysis-guidance", action="store_true", help="Ignore helper analysis guidance during proposal generation")
    parser.add_argument("--allow-external-seeds", action="store_true", help="Allow external idea seeds if wired")
    parser.add_argument("--min-viable-fill-rate", type=float, default=0.50, help="Minimum proposal fill rate before execution")
    parser.add_argument("--min-viable-candidates", type=int, default=None, help="Absolute minimum proposal candidates before execution")
    parser.add_argument("--min-large-search-candidates", type=int, default=48, help="Minimum candidates for large search proposals")
    parser.add_argument("--disable-proposal-quality-gate", action="store_true", help="Do not block underfilled proposals")
    return parser.parse_args(argv)


def _families_from_arg(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list_searchable_families()
    return [normalized]


def _workspace_root_from_base_dir(base_dir: str) -> str:
    path = Path(base_dir)
    if path.name == "experiments":
        return str(path.parent or Path("."))
    return str(path)


def _apply_runtime_memory_caps(
    batch_size: int,
    workers: int,
    decision: RuntimeDecision,
) -> tuple[int, int, dict[str, Any]]:
    """Apply memory pressure caps to batch size and worker count.

    Returns (effective_batch_size, effective_workers, caps_dict).
    caps_dict["applied"] is True if any cap was applied.
    """
    mem = current_process_memory_kb()
    rss_kb = int(mem.get("rss_kb") or 0)
    caps: dict[str, Any] = {
        "applied": False,
        "reason": None,
        "original_batch_size": batch_size,
        "original_workers": workers,
        "rss_kb": rss_kb,
    }
    if rss_kb >= _RSS_ELEVATED_KB:
        new_batch = max(8, batch_size * 2 // 3)
        new_workers = 1
        caps.update({
            "applied": True,
            "reason": "runtime_rss_pressure_elevated",
            "capped_batch_size": new_batch,
            "capped_workers": new_workers,
        })
        return new_batch, new_workers, caps
    return batch_size, workers, caps


def _load_layer2_plan():
    """Load the current Layer 2 plan. Returns (plan | None, reason_str)."""
    try:
        from planner.plan_store import load_current_plan
        result = load_current_plan()
        return result.plan if result.loaded else None, result.reason
    except Exception as exc:
        return None, f"import_error:{exc}"


def _apply_plan_hints_to_kwargs(
    kwargs: dict[str, Any],
    plan: Any,
    decision: RuntimeDecision,
    max_experiments: int,
) -> None:
    """Apply Layer 2 plan advisory hints to proposal kwargs in-place.

    Safety rules take absolute priority: confirmation and holdout decisions
    are never overridden by the plan.
    """
    structural_lane_applied = False
    if getattr(plan, "structural_execution_lane_enabled", None) is not None:
        kwargs["structural_execution_lane_enabled"] = bool(plan.structural_execution_lane_enabled)
        structural_lane_applied = True
    if getattr(plan, "structural_execution_lane_budget", None) is not None:
        lane_budget_cap = max(1, max(1, max_experiments // 3))
        kwargs["structural_execution_lane_budget"] = min(int(plan.structural_execution_lane_budget), lane_budget_cap)
        structural_lane_applied = True
    if getattr(plan, "structural_execution_lane_min_candidates", None) is not None:
        kwargs["structural_execution_lane_min_candidates"] = int(plan.structural_execution_lane_min_candidates)
        structural_lane_applied = True
    if getattr(plan, "structural_execution_lane_repeat_cap", None) is not None:
        kwargs["structural_execution_lane_repeat_cap"] = int(plan.structural_execution_lane_repeat_cap)
        structural_lane_applied = True
    if structural_lane_applied:
        print(
            "layer2_plan: structural_execution_lane="
            f"enabled={kwargs.get('structural_execution_lane_enabled', True)} "
            f"budget={kwargs.get('structural_execution_lane_budget')} "
            f"min_candidates={kwargs.get('structural_execution_lane_min_candidates')} "
            f"repeat_cap={kwargs.get('structural_execution_lane_repeat_cap')}"
        )

    if decision.confirmation_required or decision.holdout_check_required:
        print(
            f"layer2_plan_hints_skipped: confirmation_required={decision.confirmation_required} "
            f"holdout_check_required={decision.holdout_check_required} structural_lane_preserved={structural_lane_applied}"
        )
        return

    if plan.stagnation_escape_active:
        current = kwargs.get("stagnation_escape_batches", 3)
        if current > 1:
            kwargs["stagnation_escape_batches"] = 1
            print("layer2_plan: stagnation_escape_active=True → stagnation_escape_batches=1")

    if plan.repeat_branch_cap is not None and kwargs.get("repeat_branch_cap") is None:
        kwargs["repeat_branch_cap"] = plan.repeat_branch_cap
        print(f"layer2_plan: repeat_branch_cap={plan.repeat_branch_cap}")

    plan_idea_ids = list(plan.source_idea_ids or [])
    if plan_idea_ids:
        existing = list(kwargs.get("source_idea_ids") or [])
        merged = list(dict.fromkeys(existing + plan_idea_ids))
        kwargs["source_idea_ids"] = merged
        print(
            "layer2_plan: merged source_idea_ids "
            f"count={len(plan_idea_ids)} preview={_format_compact_list(plan_idea_ids)}"
        )

    if plan.families_paused:
        print(f"layer2_plan: families_paused={_format_compact_list(plan.families_paused)} (advisory only)")

    print(f"layer2_plan_applied: plan_id={plan.plan_id} rationale={_truncate_text(str(plan.rationale), max_len=120)!r}")


def _proposal_from_record(record: dict[str, Any]) -> ProposalResult:
    specs = list(record.get("candidate_specs") or [])
    families = list(record.get("strategy_families") or [])
    candidate_configs: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    candidate_metadata: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    record_source_idea_ids = list(record.get("source_idea_ids") or [])
    for spec in specs:
        family = str(spec.get("family") or "")
        params = dict(spec.get("params") or {})
        metadata = dict(spec.get("metadata") or {})
        if not family:
            continue
        if record_source_idea_ids and not metadata.get("source_idea_ids"):
            metadata["source_idea_ids"] = list(record_source_idea_ids)
        metadata.setdefault("source_planning_proposal_id", record.get("proposal_id"))
        candidate_configs.setdefault(family, []).append(params)
        candidate_metadata.setdefault(family, []).append(metadata)
    candidate_count = sum(len(items) for items in candidate_configs.values())
    quality_report = record.get("quality_report") or {}
    if not quality_report and candidate_count == 0:
        requested = sum(int(value) for value in (record.get("family_budget") or {}).values()) or 0
        quality_report = {
            "status": "fail",
            "execution_allowed": False,
            "candidate_count": 0,
            "requested": requested,
            "min_viable_candidates": 1 if requested else 0,
            "shortfall_reasons": ["empty_queued_proposal"],
        }

    request = ProposalRequest(
        proposal_id=str(record.get("proposal_id")),
        timestamp_utc=str(record.get("timestamp_utc")),
        source_batch_ids=list((record.get("analysis_provenance") or {}).get("batch_ids") or []),
        strategy_families=list(candidate_configs.keys()),
        objective_name=str(record.get("objective_name") or "wf_v1_score"),
        baseline_name=record.get("baseline_name"),
        seed=42,
        exploration_fraction=float(record.get("exploration_fraction") or 0.65),
        exploitation_fraction=float(record.get("exploitation_fraction") or 0.35),
        max_experiments=sum(len(items) for items in candidate_configs.values()),
        per_family_budgets=dict(record.get("family_budget") or {}),
        resume=True,
        source_idea_ids=record_source_idea_ids,
    )
    return ProposalResult(
        request=request,
        status=str(record.get("status") or "pending_execution"),
        candidate_configs=candidate_configs,
        reasoning_summary={
            "planning_rationale": record.get("planning_rationale") or {},
            "analysis_provenance": record.get("analysis_provenance") or {},
            "proposal_quality": quality_report,
            "source": "planning_agent_queue",
        },
        candidate_metadata=candidate_metadata,
        proposal_path=None,
        summary_path=None,
    )


def _proposal_execution_allowed(proposal: ProposalResult) -> bool:
    quality = (proposal.reasoning_summary or {}).get("proposal_quality") or {}
    if not quality:
        return True
    return bool(quality.get("execution_allowed", True))


def _print_proposal_quality(proposal: ProposalResult) -> None:
    quality = (proposal.reasoning_summary or {}).get("proposal_quality") or {}
    if not quality:
        return
    print(f"proposal_quality_status={quality.get('status')}")
    print(f"proposal_quality_score={quality.get('quality_score')}")
    print(
        "proposal_quality_fill="
        f"{quality.get('candidate_count')}/{quality.get('requested')} "
        f"min_viable={quality.get('min_viable_candidates')}"
    )
    if quality.get("shortfall_reasons"):
        print(f"proposal_shortfall_reasons={_format_compact_list(quality.get('shortfall_reasons'))}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    families = _families_from_arg(args.family)
    workspace_root = _workspace_root_from_base_dir(args.base_dir)

    # Load Layer 2 plan (advisory only, always fails gracefully)
    layer2_plan, plan_reason = _load_layer2_plan()
    if layer2_plan is not None:
        print(f"layer2_plan_loaded: plan_id={layer2_plan.plan_id} valid_until={layer2_plan.valid_until}")
    else:
        print(f"layer2_plan: {plan_reason}")

    queued_record = None
    consumed_proposal_id: str | None = None
    if args.run_proposal and not args.proposal_next:
        queued_record = load_latest_pending_proposal_record(workspace_root)

    if args.proposal_next:
        # Build runtime decision from live experiment state
        decision = build_runtime_decision(RuntimeDecisionInput(
            workspace_root=workspace_root,
            experiments_dir=args.base_dir,
            strategy_families=families,
            max_experiments=args.n,
            seed=args.seed,
            exploration_fraction=args.exploration_fraction,
            exploitation_fraction=args.exploitation_fraction,
            stagnation_escape_batches=args.stagnation_escape_batches,
            min_large_search_candidates=args.min_large_search_candidates,
        ))
        print(f"runtime_decision_id={decision.decision_id}")
        print(f"runtime_decision_cycle_mode={decision.cycle_mode}")
        print(f"runtime_decision_families={_format_selected_families(decision.selected_families)}")
        print(f"runtime_decision_fallback={decision.fallback_used}")

        # Apply memory pressure caps
        effective_n, effective_workers, mem_caps = _apply_runtime_memory_caps(
            args.n, args.max_workers, decision
        )
        if mem_caps.get("applied"):
            print(f"runtime_memory_caps_applied: {_format_compact_dict(mem_caps)}")

        # Determine active families (runtime decision overrides CLI arg when available)
        active_families = decision.selected_families if decision.selected_families else families

        # Base exploration/exploitation from runtime decision (or fall back to CLI args)
        if not decision.fallback_used:
            exploration_fraction = decision.exploration_fraction
            exploitation_fraction = decision.exploitation_fraction
        else:
            exploration_fraction = args.exploration_fraction
            exploitation_fraction = args.exploitation_fraction

        # Build proposal kwargs from runtime decision
        proposal_kwargs: dict[str, Any] = dict(
            strategy_families=active_families,
            seed=args.seed,
            exploration_fraction=exploration_fraction,
            exploitation_fraction=exploitation_fraction,
            max_experiments=effective_n,
            resume=not args.no_resume,
            novelty_floor=args.novelty_floor,
            template_fraction=args.template_fraction,
            cross_family_fraction=args.cross_family_fraction,
            max_near_duplicate_distance=args.max_near_duplicate_distance,
            stagnation_escape_batches=args.stagnation_escape_batches,
            allow_external_seeds=args.allow_external_seeds,
            use_idea_queue=not args.no_helper_ideas,
            use_analysis_guidance=not args.no_analysis_guidance,
            min_viable_fill_rate=args.min_viable_fill_rate,
            min_viable_candidates=args.min_viable_candidates,
            min_large_search_candidates=decision.min_large_search_candidates,
            quality_gate=not args.disable_proposal_quality_gate,
            structural_novelty_threshold=decision.structural_novelty_threshold,
            history_limit_per_family=max(1, effective_n // 2),
        )

        if not decision.fallback_used:
            proposal_kwargs.update(dict(
                per_family_budgets=decision.family_budgets,
                new_idea_quota=decision.new_idea_quota,
                uncommon_idea_quota=decision.uncommon_idea_quota,
                repeat_branch_cap=decision.repeat_branch_cap,
                max_same_template_per_cycle=decision.max_same_template_per_cycle,
                max_same_lineage_per_cycle=decision.max_same_lineage_per_cycle,
                branch_budgets=decision.branch_budgets,
                branch_budget_rationale=decision.branch_budget_rationale,
            ))

        # Handle confirmation cycle (overrides normal budget/fraction settings)
        if decision.confirmation_required:
            planned_n = decision.planned_max_experiments or effective_n
            proposal_kwargs.update(dict(
                max_experiments=planned_n,
                per_family_budgets=decision.confirmation_family_budgets or decision.family_budgets,
                confirmation_state=decision.promotion_state,
                confirmation_required=True,
                confirmation_reason=decision.confirmation_reason,
                confirmation_batch_id=decision.confirmation_batch_id,
                confirmation_focus_family=decision.winner_family,
                confirmation_batch_experiments=planned_n,
                history_limit_per_family=planned_n,
            ))
            print(f"runtime_decision: confirmation_required batch_id={decision.confirmation_batch_id}")

        # Handle targeted follow-up
        if decision.targeted_follow_up_required:
            proposal_kwargs.update(dict(
                targeted_follow_up_required=True,
                targeted_follow_up_reason=decision.targeted_follow_up_reason,
                targeted_follow_up_type=decision.targeted_follow_up_type,
                targeted_follow_up_priority=decision.targeted_follow_up_priority or 0.0,
                targeted_follow_up_batch_id=decision.targeted_follow_up_batch_id,
            ))

        # Handle holdout check
        if decision.holdout_check_required:
            proposal_kwargs.update(dict(
                holdout_check_required=True,
                holdout_check_type=decision.holdout_check_type,
                holdout_check_status=decision.holdout_check_status,
                holdout_check_outcome=decision.holdout_check_outcome,
                holdout_check_scope=decision.holdout_check_scope,
                holdout_check_batch_id=decision.holdout_check_batch_id,
                holdout_horizon_tags=decision.holdout_horizon_tags,
                holdout_regime_tags=decision.holdout_regime_tags,
            ))

        # Apply Layer 2 plan hints (advisory — safety rules already applied above)
        if layer2_plan is not None:
            _apply_plan_hints_to_kwargs(proposal_kwargs, layer2_plan, decision, effective_n)

        proposal_request = build_proposal_request(**proposal_kwargs)
        proposal = generate_next_round_proposal(proposal_request, base_dir=args.base_dir)

        # Enrich reasoning summary with runtime decision (for batch metadata lineage)
        enriched_summary = dict(proposal.reasoning_summary or {})
        enriched_summary["runtime_decision"] = asdict(decision)
        if layer2_plan is not None:
            enriched_summary["layer2_plan_id"] = layer2_plan.plan_id
            enriched_summary["layer2_plan"] = {
                "plan_id": layer2_plan.plan_id,
                "structural_execution_lane_enabled": layer2_plan.structural_execution_lane_enabled,
                "structural_execution_lane_budget": layer2_plan.structural_execution_lane_budget,
                "structural_execution_lane_min_candidates": layer2_plan.structural_execution_lane_min_candidates,
                "structural_execution_lane_repeat_cap": layer2_plan.structural_execution_lane_repeat_cap,
                "repeat_branch_cap": layer2_plan.repeat_branch_cap,
                "source_idea_ids": list(layer2_plan.source_idea_ids or []),
            }
        proposal = dc_replace(proposal, reasoning_summary=enriched_summary)

        print(f"proposal_id={proposal.request.proposal_id}")
        print(f"proposal_status={proposal.status}")
        print(f"proposal_reasoning={_format_reasoning_summary(proposal.reasoning_summary)}")
        _print_proposal_quality(proposal)
        if proposal.proposal_path:
            print(f"proposal={proposal.proposal_path}")
        if proposal.summary_path:
            print(f"proposal_summary={proposal.summary_path}")

        if not args.run_proposal:
            return 0
        if not _proposal_execution_allowed(proposal):
            print("proposal_execution=blocked_by_quality_gate")
            return 0

        request = proposal_to_batch_request(
            proposal,
            persist=not args.no_persist,
            resume=not args.no_resume,
            max_workers=effective_workers,
        )
        # Attach runtime_decision to batch request metadata for provenance
        req_metadata = dict(request.proposal_metadata or {})
        req_metadata["runtime_decision"] = enriched_summary["runtime_decision"]
        request = dc_replace(request, proposal_metadata=req_metadata)

    elif queued_record is not None:
        proposal = _proposal_from_record(queued_record)
        print(f"proposal_id={proposal.request.proposal_id}")
        print("proposal_status=queued_planning_proposal")
        print(f"proposal_reasoning={_format_reasoning_summary(proposal.reasoning_summary)}")
        _print_proposal_quality(proposal)
        if not _proposal_execution_allowed(proposal):
            print("proposal_execution=blocked_by_quality_gate")
            update_proposal_record_status(
                proposal.request.proposal_id,
                status="quality_blocked",
                workspace_root=workspace_root,
            )
            return 0
        request = proposal_to_batch_request(
            proposal,
            persist=not args.no_persist,
            resume=not args.no_resume,
            max_workers=args.max_workers,
        )
        consumed_proposal_id = proposal.request.proposal_id
    elif args.run_proposal:
        proposal_request = build_proposal_request(
            strategy_families=families,
            seed=args.seed,
            exploration_fraction=args.exploration_fraction,
            exploitation_fraction=args.exploitation_fraction,
            max_experiments=args.n,
            resume=not args.no_resume,
            novelty_floor=args.novelty_floor,
            template_fraction=args.template_fraction,
            cross_family_fraction=args.cross_family_fraction,
            max_near_duplicate_distance=args.max_near_duplicate_distance,
            stagnation_escape_batches=args.stagnation_escape_batches,
            allow_external_seeds=args.allow_external_seeds,
            use_idea_queue=not args.no_helper_ideas,
            use_analysis_guidance=not args.no_analysis_guidance,
            min_viable_fill_rate=args.min_viable_fill_rate,
            min_viable_candidates=args.min_viable_candidates,
            min_large_search_candidates=args.min_large_search_candidates,
            quality_gate=not args.disable_proposal_quality_gate,
        )
        proposal = generate_next_round_proposal(proposal_request, base_dir=args.base_dir)
        print(f"proposal_id={proposal.request.proposal_id}")
        print(f"proposal_status={proposal.status}")
        print(f"proposal_reasoning={_format_reasoning_summary(proposal.reasoning_summary)}")
        _print_proposal_quality(proposal)
        if proposal.proposal_path:
            print(f"proposal={proposal.proposal_path}")
        if proposal.summary_path:
            print(f"proposal_summary={proposal.summary_path}")
        if not _proposal_execution_allowed(proposal):
            print("proposal_execution=blocked_by_quality_gate")
            return 0
        request = proposal_to_batch_request(
            proposal,
            persist=not args.no_persist,
            resume=not args.no_resume,
            max_workers=args.max_workers,
        )
    else:
        request = build_batch_request(
            strategy_families=families,
            sampler_type=args.method,
            max_experiments=args.n,
            max_per_family=args.max_per_family,
            seed=args.seed,
            persist=not args.no_persist,
            resume=not args.no_resume,
            max_workers=args.max_workers,
        )

    batch_result = run_batch_experiments(request, base_dir=args.base_dir)

    if consumed_proposal_id:
        update_proposal_record_status(
            consumed_proposal_id,
            status="consumed",
            workspace_root=workspace_root,
        )

    print(f"batch_id={batch_result.request.batch_id}")
    print(
        f"sampled={batch_result.total_sampled} executed={batch_result.total_executed} "
        f"skipped={batch_result.total_skipped} failed={batch_result.total_failed}"
    )
    if batch_result.total_executed < batch_result.request.max_experiments:
        print(
            f"requested={batch_result.request.max_experiments} sampled={batch_result.total_sampled} "
            f"executed={batch_result.total_executed} shortfall={batch_result.request.max_experiments - batch_result.total_executed}"
        )
    leaderboard = build_batch_leaderboard(batch_result.results)
    if not leaderboard.empty:
        print(_format_leaderboard_preview(leaderboard, max_rows=_LEADERBOARD_PREVIEW_ROWS))
    if batch_result.leaderboard_path:
        print(f"leaderboard={batch_result.leaderboard_path}")
    if batch_result.summary_path:
        print(f"summary={batch_result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
