#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agents.schemas import load_latest_pending_proposal_record, update_proposal_record_status

from experiment_batch import (
    build_batch_leaderboard,
    build_batch_request,
    proposal_to_batch_request,
    run_batch_experiments,
)
from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_spaces import list_searchable_families
from experiment_types import ProposalRequest, ProposalResult


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


def _proposal_from_record(record: dict[str, Any]) -> ProposalResult:
    specs = list(record.get("candidate_specs") or [])
    families = list(record.get("strategy_families") or [])
    candidate_configs: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    candidate_metadata: dict[str, list[dict[str, Any]]] = {family: [] for family in families}
    for spec in specs:
        family = str(spec.get("family") or "")
        params = dict(spec.get("params") or {})
        metadata = dict(spec.get("metadata") or {})
        if not family:
            continue
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
        source_idea_ids=list(record.get("source_idea_ids") or []),
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
        print(f"proposal_shortfall_reasons={quality.get('shortfall_reasons')}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    families = _families_from_arg(args.family)
    workspace_root = _workspace_root_from_base_dir(args.base_dir)
    queued_record = None
    consumed_proposal_id: str | None = None
    if args.run_proposal and not args.proposal_next:
        queued_record = load_latest_pending_proposal_record(workspace_root)

    if args.proposal_next:
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
        print(proposal.reasoning_summary)
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
            max_workers=args.max_workers,
        )
    elif queued_record is not None:
        proposal = _proposal_from_record(queued_record)
        print(f"proposal_id={proposal.request.proposal_id}")
        print("proposal_status=queued_planning_proposal")
        print(proposal.reasoning_summary)
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
        print(proposal.reasoning_summary)
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
        print(leaderboard.head(10).to_string(index=False))
    if batch_result.leaderboard_path:
        print(f"leaderboard={batch_result.leaderboard_path}")
    if batch_result.summary_path:
        print(f"summary={batch_result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
