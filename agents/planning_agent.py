#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.schemas import (
    ProposalRecord,
    ensure_helper_dirs,
    load_idea_records,
    load_latest_analysis_report,
    save_proposal_record,
)
from experiment_refinement import build_proposal_request, generate_next_round_proposal
from experiment_spaces import list_searchable_families


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only helper that writes proposal records")
    parser.add_argument("--workspace-root", default=".", help="Repo root for helper queues/reports")
    parser.add_argument("--experiments-dir", default="experiments", help="Official experiment store to read")
    parser.add_argument("--family", default="all", help="Specific family or all")
    parser.add_argument("--n", type=int, default=24, help="Maximum candidate experiments to plan")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exploration-fraction", type=float, default=0.65, help="Proposal exploration fraction")
    parser.add_argument("--exploitation-fraction", type=float, default=0.35, help="Proposal exploitation fraction")
    parser.add_argument("--min-viable-fill-rate", type=float, default=0.50, help="Minimum proposal fill rate before execution")
    parser.add_argument("--min-large-search-candidates", type=int, default=48, help="Minimum candidates for large-search proposals")
    parser.add_argument("--no-helper-ideas", action="store_true", help="Ignore queued helper ideas during planning")
    parser.add_argument("--no-analysis-guidance", action="store_true", help="Ignore analysis guidance during planning")
    return parser.parse_args(argv)


def _families(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list_searchable_families()
    return [normalized]


def build_planning_proposal(
    *,
    workspace_root: str = ".",
    experiments_dir: str = "experiments",
    families: list[str] | None = None,
    max_experiments: int = 24,
    seed: int = 42,
    exploration_fraction: float = 0.65,
    exploitation_fraction: float = 0.35,
    min_viable_fill_rate: float = 0.50,
    min_large_search_candidates: int = 48,
    use_idea_queue: bool = True,
    use_analysis_guidance: bool = True,
) -> ProposalRecord:
    ensure_helper_dirs(workspace_root)
    selected_families = families or list_searchable_families()
    ideas = load_idea_records(workspace_root) if use_idea_queue else []
    latest_analysis = load_latest_analysis_report(workspace_root) if use_analysis_guidance else None

    request = build_proposal_request(
        strategy_families=selected_families,
        max_experiments=max_experiments,
        seed=seed,
        exploration_fraction=exploration_fraction,
        exploitation_fraction=exploitation_fraction,
        use_idea_queue=use_idea_queue,
        use_analysis_guidance=use_analysis_guidance,
        min_viable_fill_rate=min_viable_fill_rate,
        min_large_search_candidates=min_large_search_candidates,
    )
    proposal = generate_next_round_proposal(request, base_dir=experiments_dir)

    candidate_specs: list[dict[str, object]] = []
    family_budget: dict[str, int] = {}
    for family, configs in proposal.candidate_configs.items():
        metadata_items = (proposal.candidate_metadata or {}).get(family, [])
        family_budget[family] = len(configs)
        for idx, config in enumerate(configs):
            metadata = metadata_items[idx] if idx < len(metadata_items) else {}
            candidate_specs.append(
                {
                    "family": family,
                    "params": config,
                    "metadata": metadata,
                }
            )

    idea_ids = list(proposal.request.source_idea_ids or [])
    planning_rationale = {
        "reasoning_summary": proposal.reasoning_summary,
        "helper_idea_count": len(ideas),
        "analysis_guidance_used": latest_analysis is not None,
    }
    analysis_provenance = None
    if latest_analysis:
        analysis_provenance = {
            "report_id": latest_analysis.get("report_id"),
            "batch_ids": latest_analysis.get("batch_ids", []),
            "next_focus": latest_analysis.get("next_focus", []),
        }

    return ProposalRecord(
        proposal_id=proposal.request.proposal_id,
        strategy_families=list(proposal.request.strategy_families),
        source_idea_ids=idea_ids,
        candidate_specs=candidate_specs,
        exploration_fraction=proposal.request.exploration_fraction,
        exploitation_fraction=proposal.request.exploitation_fraction,
        family_budget=family_budget,
        timestamp_utc=proposal.request.timestamp_utc,
        objective_name=proposal.request.objective_name,
        baseline_name=proposal.request.baseline_name,
        planning_rationale=planning_rationale,
        analysis_provenance=analysis_provenance,
        quality_report=proposal.reasoning_summary.get("proposal_quality"),
        metadata={
            "generated_by": "planning_agent",
            "helper_idea_ids_seen": [record.get("idea_id") for record in ideas if record.get("idea_id")],
            "proposal_status": proposal.status,
            "proposal_path": proposal.proposal_path,
            "summary_path": proposal.summary_path,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = build_planning_proposal(
        workspace_root=args.workspace_root,
        experiments_dir=args.experiments_dir,
        families=_families(args.family),
        max_experiments=args.n,
        seed=args.seed,
        exploration_fraction=args.exploration_fraction,
        exploitation_fraction=args.exploitation_fraction,
        min_viable_fill_rate=args.min_viable_fill_rate,
        min_large_search_candidates=args.min_large_search_candidates,
        use_idea_queue=not args.no_helper_ideas,
        use_analysis_guidance=not args.no_analysis_guidance,
    )
    path = save_proposal_record(record, workspace_root=args.workspace_root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
